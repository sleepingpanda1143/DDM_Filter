#!/usr/bin/env python3
"""
Train and evaluate GluonTS-style TimeGrad (diffusers + pts TimeGradEstimator) on
pre-built dataset folders that contain ``preprocess_metadata.json`` (radar / motion runs).

Example:
  ./scripts/run_timegrad.sh smoke
  ./scripts/run_timegrad.sh train --preset radar_viz_debug --max-epochs 20
  TIMEGRAD_CUDA_DEVICE=0 ./scripts/run_timegrad.sh train ...   # 仅用物理 GPU 0

**Metrics semantics (read before comparing runs):**

- ``train_loss`` is the **diffusion / noise-prediction loss inside the model’s *scaled* working space**
  (after GluonTS ``MeanScaler`` / ``StdScaler`` when ``scaling!=none``). Its **absolute value is not comparable**
  across ``--timegrad-scaling`` (none vs mean vs std) or across different **ListDataset** target preprocessing
  (absolute vs Δ-step), because both the loss definition and typical magnitude of tensors change.
- ``ADE_xy`` in logs is computed on **whatever channels 0–1 are in the evaluation tensor**. With
  ``delta_step_increment_*``, those are **per-step Δx, Δy in metres**, i.e. error on **increments**, not the
  same as **absolute planar position error** along the full path (use ``error_comparison_timegrad.png`` /
  NPZ-reconstructed RMSE for that).

After ``train`` / ``smoke``, for ``target_dim >= 2`` saves:

- ``position_step_eval.png`` — pooled ADE/FDE style stats in **model target units** (see spatial tag).
- ``trajectories_global_test_xy.png`` — all test GT paths in absolute (x,y) from NPZ.
- ``trajectory_detail_panel.png`` — first N test series: GT vs median forecast path (absolute plane).
- ``trajectory_full_and_tail_panel.png`` — same series as detail: **left = full** absolute (x,y);
  **right = tight end zoom** (default: last ``prediction_length + --eval-trajectory-forecast-context-steps``
  steps, axis limits from data; use ``--eval-trajectory-right-panel=fixed_tail`` for a plain tail window).
  Draws **all sample paths** (faint) with **mean** and **median** overlays.
- ``error_comparison_timegrad.png`` — 2×2 layout aligned with ``Radar/examples/particle_filter_example.py``
  (trajectory | pos error vs time | vel error vs time | RMSE bars), using **GT vs TimeGrad median**.

Use ``--no-eval-plots`` to skip all of the above.

Default **terminal** line keeps only MASE, **absolute-plane forecast-window** pooled ``RMSE_xy``
and mean ``ADE_xy`` (metres), plus ``train_s``; full GluonTS / increment-space diagnostics stay in
``summary.json``. Use ``--verbose-eval-metrics`` for the previous verbose console dump.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from diffusers import DDPMScheduler
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.time_feature import get_lags_for_frequency

REPO_ROOT = Path(__file__).resolve().parents[1]


def _required_past_length(freq: str, context_length: int) -> int:
    lags = [x - 1 for x in get_lags_for_frequency(freq_str=freq)]
    return int(context_length + max(lags))


def _load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    path = dataset_dir / "preprocess_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with path.open() as f:
        return json.load(f)


def _resolved_delta_encoding(meta: Dict[str, Any]) -> str:
    """How ``state_repr=delta`` + ``position_origin=first`` maps NPZ rows to model targets."""
    raw = meta.get("delta_encoding")
    if raw is None or str(raw).strip() == "":
        # Default matches legacy RNN stats: ``normalization_std`` ~ O(10²) m is **per-step**
        # motion, not ``x(t)-x(0)`` (which can stay ~10⁴ m for long flights).
        return "step_increment"
    return str(raw).lower().replace("-", "_")


def _spatial_transform_label(meta: Dict[str, Any]) -> tuple[str, bool]:
    """
    Human-readable tag for logs/summary, and whether dims 0–1 are **per-step Δx,Δy**
    (vs absolute plane coordinates in m).
    """
    sr = str(meta.get("state_repr", "absolute")).lower()
    po = str(meta.get("position_origin", "none")).lower()
    if sr == "delta" and po == "first":
        de = _resolved_delta_encoding(meta)
        if de in ("step_increment", "step", "diff"):
            return "delta_step_increment_row0_zero", True
        if de in (
            "cumulative_from_first",
            "cumulative",
            "origin",
            "minus_first",
            "delta_minus_first",
        ):
            return "delta_cumulative_from_first_row", False
        return f"delta_encoding_{de}", False
    if sr == "delta":
        return "delta_as_in_npz", False
    return "absolute_as_in_npz", False


def _trajectory_to_model_targets(arr_in: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """
    Spatial representation aligned with legacy radar presets (RNN TimeGrad runs).

    - ``state_repr=absolute`` / ``position_origin=none``: use ``arr`` as stored (typical raw NPZ).
    - ``state_repr=delta`` + ``position_origin=first`` (NPZ still in absolute coordinates):

      * Default (``delta_encoding`` omitted): **step increments** — row ``t>0`` is
        ``arr[t]-arr[t-1]``; row ``0`` is zeros. Matches ``normalization_std`` in preset
        metadata (order ~10² m per step), unlike ``arr-arr[0]`` which keeps ~10⁴ m offsets.
      * ``delta_encoding: cumulative_from_first`` (aliases ``cumulative``, ``minus_first``):
        ``arr - arr[0]`` (cumulative displacement from trajectory start; can be ill-conditioned
        with ``timegrad_scaling: false``).

    If the NPZ is already stored as deltas, set ``position_origin=none`` so no transform runs.

    This path is **independent of LSTM vs Transformer**; only the old experiment scripts used
    to apply delta before training while this loader originally did not.
    """
    arr = np.array(arr_in, dtype=np.float64, copy=True)
    state_repr = str(meta.get("state_repr", "absolute")).lower()
    origin = str(meta.get("position_origin", "none")).lower()
    if state_repr == "delta":
        if origin == "first":
            de = _resolved_delta_encoding(meta)
            if de in ("step_increment", "step", "diff"):
                out = np.zeros_like(arr)
                out[1:] = arr[1:] - arr[:-1]
                return out
            if de in (
                "cumulative_from_first",
                "cumulative",
                "origin",
                "minus_first",
                "delta_minus_first",
            ):
                return arr - arr[0:1, :]
            raise ValueError(
                f"Unsupported delta_encoding={de!r} for state_repr=delta position_origin=first "
                f"(use 'step_increment' or 'cumulative_from_first')."
            )
        elif origin not in ("none", ""):
            raise ValueError(
                f"Unsupported position_origin={origin!r} for state_repr=delta "
                f"(use 'first' for NPZ in absolute coords, or 'none' if NPZ is already delta)."
            )
    elif state_repr not in ("absolute", "origin", ""):
        raise ValueError(f"Unsupported state_repr={state_repr!r}")
    return arr


def _apply_metadata_zscore_channels(arr: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """
    Per-channel affine from ``preprocess_metadata.json`` (same length as ``target_dim``).

    Applied **after** spatial transforms (e.g. step increments). Ensure ``normalization_mean`` /
    ``normalization_std`` were estimated on a compatible representation, or treat as an ablation only.
    """
    d = int(meta["target_dim"])
    mean = np.asarray(meta.get("normalization_mean"), dtype=np.float64).reshape(-1)
    std = np.asarray(meta.get("normalization_std"), dtype=np.float64).reshape(-1)
    if mean.size < d or std.size < d:
        raise ValueError(
            f"normalization_mean/std must have length >= target_dim={d} "
            f"(got mean={mean.size}, std={std.size})"
        )
    mean = mean[:d].reshape(1, -1)
    std = np.maximum(std[:d].reshape(1, -1), 1e-8)
    x = np.asarray(arr, dtype=np.float64)
    return (x - mean) / std


def _denorm_metadata_zscore_channels(td: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`_apply_metadata_zscore_channels` for ``target_dim`` columns."""
    d = int(meta["target_dim"])
    mean = np.asarray(meta.get("normalization_mean"), dtype=np.float64).reshape(-1)[:d]
    std = np.asarray(meta.get("normalization_std"), dtype=np.float64).reshape(-1)[:d]
    t = np.asarray(td, dtype=np.float64)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    return t * std.reshape(1, -1) + mean.reshape(1, -1)


def _build_list_dataset(
    meta: Dict[str, Any],
    indices: List[int],
    min_length: int,
    *,
    apply_metadata_zscore: bool = False,
) -> tuple[ListDataset, List[int]]:
    npz_path = Path(meta["input_npz"])
    if not npz_path.is_file():
        raise FileNotFoundError(f"input_npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "trajectories" not in data.files:
        raise KeyError(f"{npz_path} has no 'trajectories' array ({data.files})")

    trajs = data["trajectories"]
    freq = str(meta["freq"])
    entries = []
    traj_ids: List[int] = []
    for idx in indices:
        arr = np.asarray(trajs[int(idx)], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != int(meta["target_dim"]):
            raise ValueError(
                f"Trajectory {idx} has shape {arr.shape}, expected (T, {meta['target_dim']})"
            )
        if arr.shape[0] < min_length:
            continue
        # Uniform length so stacked batches are rectangular (variable-length series
        # otherwise break ``as_stacked_batches``).
        arr = arr[:min_length]
        arr = _trajectory_to_model_targets(arr, meta)
        if apply_metadata_zscore:
            arr = _apply_metadata_zscore_channels(arr, meta)
        d = int(meta["target_dim"])
        # GluonTS multivariate convention: ``target`` is (target_dim, time), not (time, target_dim).
        if d > 1:
            tgt = arr.T.copy()
        else:
            tgt = np.asarray(arr, dtype=np.float64).reshape(-1)
        entries.append(
            {
                "start": pd.Period("2020-01-01", freq=freq),
                "target": tgt,
            }
        )
        traj_ids.append(int(idx))
    if not entries:
        raise RuntimeError(
            f"No series left after min_length={min_length} filter "
            f"(check trajectories vs context/lags for freq={freq})."
        )
    ds = ListDataset(entries, freq=freq, one_dim_target=(int(meta["target_dim"]) == 1))
    return ds, traj_ids


def _metrics_for_json(agg_metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in agg_metrics.items():
        try:
            fv = float(np.asarray(v).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(fv):
            out[k] = fv
    return out


def _collect_aligned_pred_obs(
    forecasts: List[Any],
    tss: List[pd.DataFrame],
    prediction_length: int,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    *,
    use_global_affine: bool,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """Median forecast vs ground truth, same units as ListDataset (``(T_pred, D)``)."""
    if use_global_affine and (mean is None or std is None):
        return []
    mean_a = np.asarray(mean, dtype=np.float64).reshape(-1) if mean is not None else None
    std_a = np.asarray(std, dtype=np.float64).reshape(-1) if std is not None else None
    pairs: List[tuple[np.ndarray, np.ndarray]] = []
    for fc, ts in zip(forecasts, tss):
        pred = np.asarray(fc.quantile(0.5), dtype=np.float64)
        obs = np.asarray(ts.to_numpy(copy=False), dtype=np.float64)
        obs = obs[-prediction_length:]
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        if pred.shape != obs.shape:
            raise ValueError(
                f"Forecast shape {pred.shape} != observation shape {obs.shape} "
                f"(check multivariate layout / prediction_length)."
            )
        if use_global_affine and mean_a is not None and std_a is not None:
            p = pred * std_a + mean_a
            o = obs * std_a + mean_a
        else:
            p, o = pred, obs
        pairs.append((p, o))
    return pairs


def _position_velocity_eval(
    pairs: List[tuple[np.ndarray, np.ndarray]],
    target_dim: int,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Radar-style **planar position** error (dims 0,1 = x,y in m) and optional velocity (2:4).

    ADE: mean over forecast horizon of ||pred_xy - gt_xy||_2 per series; report mean/median across series.
    FDE: same norm at last forecast step (equals ADE when prediction_length is 1).
    """
    if target_dim < 2:
        return {}, {}

    ades: List[float] = []
    fdes: List[float] = []
    vel_scalars: List[float] = []
    all_ex: List[float] = []
    all_ey: List[float] = []

    for p, o in pairs:
        ex = (p[:, 0] - o[:, 0]).ravel()
        ey = (p[:, 1] - o[:, 1]).ravel()
        all_ex.extend(ex.tolist())
        all_ey.extend(ey.tolist())
        epos = np.sqrt((p[:, 0] - o[:, 0]) ** 2 + (p[:, 1] - o[:, 1]) ** 2)
        ades.append(float(np.mean(epos)))
        fdes.append(float(epos[-1]))
        if target_dim >= 4:
            vel_scalars.append(
                float(np.sqrt(np.mean((p[:, 2:4] - o[:, 2:4]) ** 2)))
            )

    ex_arr = np.asarray(all_ex, dtype=np.float64)
    ey_arr = np.asarray(all_ey, dtype=np.float64)
    stats: Dict[str, float] = {
        "ade_xy_mean_m": float(np.mean(ades)),
        "ade_xy_median_m": float(np.median(ades)),
        "fde_xy_mean_m": float(np.mean(fdes)),
        "fde_xy_median_m": float(np.median(fdes)),
        "rmse_x_m": float(np.sqrt(np.mean(ex_arr**2))),
        "rmse_y_m": float(np.sqrt(np.mean(ey_arr**2))),
    }
    if vel_scalars:
        stats["rmse_vel_plane_ms"] = float(np.mean(vel_scalars))
        stats["rmse_vel_plane_median_ms"] = float(np.median(vel_scalars))

    arrays = {
        "per_series_ade_m": np.asarray(ades, dtype=np.float64),
        "per_series_fde_m": np.asarray(fdes, dtype=np.float64),
        "per_series_vel_rmse_ms": np.asarray(vel_scalars, dtype=np.float64) if vel_scalars else np.array([]),
    }
    return stats, arrays


def _save_position_eval_figure(
    out_dir: Path,
    stats: Dict[str, float],
    arrays: Dict[str, np.ndarray],
    preset_name: str,
    *,
    ch01_step_increments: bool = False,
) -> Path:
    """2×2 figure similar in spirit to Radar ``error_comparison`` / PF example plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sem = (
        "channels 0–1 = per-step Δx,Δy (m); not cumulative absolute position"
        if ch01_step_increments
        else "channels 0–1 = planar x,y (m)"
    )
    fig.suptitle(
        f"TimeGrad median forecast — {sem} ({preset_name})",
        fontsize=13,
    )

    ades = arrays.get("per_series_ade_m", np.array([]))
    ax0 = axes[0, 0]
    if ades.size:
        ax0.hist(ades, bins=min(40, max(10, len(ades) // 3)), color="steelblue", edgecolor="white", alpha=0.9)
        ax0.axvline(
            stats.get("ade_xy_median_m", float("nan")),
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"median ADE_xy = {stats.get('ade_xy_median_m', float('nan')):.2f} m",
        )
        ax0.axvline(
            stats.get("ade_xy_mean_m", float("nan")),
            color="darkorange",
            linestyle=":",
            linewidth=2,
            label=f"mean ADE_xy = {stats.get('ade_xy_mean_m', float('nan')):.2f} m",
        )
    ax0.set_xlabel(
        "ADE_Δxy per series (m)" if ch01_step_increments else "ADE_xy per series (m)"
    )
    ax0.set_ylabel("Count")
    ax0.set_title(
        "Distribution of mean per-step planar increment error"
        if ch01_step_increments
        else "Distribution of mean planar position error over forecast horizon"
    )
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[0, 1]
    fdes = arrays.get("per_series_fde_m", np.array([]))
    if fdes.size:
        ax1.hist(fdes, bins=min(40, max(10, len(fdes) // 3)), color="seagreen", edgecolor="white", alpha=0.9)
        ax1.axvline(
            stats.get("fde_xy_median_m", float("nan")),
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"median FDE_xy = {stats.get('fde_xy_median_m', float('nan')):.2f} m",
        )
        ax1.axvline(
            stats.get("fde_xy_mean_m", float("nan")),
            color="darkorange",
            linestyle=":",
            linewidth=2,
            label=f"mean FDE_xy = {stats.get('fde_xy_mean_m', float('nan')):.2f} m",
        )
    ax1.set_xlabel(
        "FDE_Δxy per series (m)" if ch01_step_increments else "FDE_xy per series (m)"
    )
    ax1.set_ylabel("Count")
    ax1.set_title("Final-step planar error (same as ADE when horizon is 1)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1, 0]
    vels = arrays.get("per_series_vel_rmse_ms", np.array([]))
    if vels.size:
        ax2.hist(vels, bins=min(40, max(10, len(vels) // 3)), color="mediumpurple", edgecolor="white", alpha=0.9)
        ax2.axvline(
            stats.get("rmse_vel_plane_median_ms", float("nan")),
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"median RMSE_v = {stats.get('rmse_vel_plane_median_ms', float('nan')):.3f} m/s",
        )
        ax2.axvline(
            stats.get("rmse_vel_plane_ms", float("nan")),
            color="darkorange",
            linestyle=":",
            linewidth=2,
            label=f"mean RMSE_v = {stats.get('rmse_vel_plane_ms', float('nan')):.3f} m/s",
        )
        ax2.set_xlabel("Per-series RMSE of (vx, vy) over horizon (m/s)")
        ax2.set_ylabel("Count")
        ax2.set_title("Velocity error (channels 2–3)")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "target_dim < 4: no velocity channels", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    ax3 = axes[1, 1]
    ax3.axis("off")
    xy_hdr = (
        "Pooled RMSE on Δx,Δy increments (all test points, horizon):"
        if ch01_step_increments
        else "Pooled RMSE (all test points, horizon):"
    )
    lines = [
        xy_hdr,
        f"  RMSE_x = {stats.get('rmse_x_m', float('nan')):.3f} m",
        f"  RMSE_y = {stats.get('rmse_y_m', float('nan')):.3f} m",
        "",
        "Per-series aggregates:",
        f"  ADE_xy mean / median = {stats.get('ade_xy_mean_m', float('nan')):.3f} / "
        f"{stats.get('ade_xy_median_m', float('nan')):.3f} m",
        f"  FDE_xy mean / median = {stats.get('fde_xy_mean_m', float('nan')):.3f} / "
        f"{stats.get('fde_xy_median_m', float('nan')):.3f} m",
    ]
    if "rmse_vel_plane_ms" in stats:
        lines.extend(
            [
                f"  RMSE (vx,vy) mean / median = {stats['rmse_vel_plane_ms']:.4f} / "
                f"{stats.get('rmse_vel_plane_median_ms', float('nan')):.4f} m/s",
            ]
        )
    ax3.text(0.05, 0.95, "\n".join(lines), transform=ax3.transAxes, fontsize=11, va="top", family="monospace")

    out_path = out_dir / "position_step_eval.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _ts_to_time_target(ts: pd.DataFrame, target_dim: int) -> np.ndarray:
    arr = np.asarray(ts.to_numpy(copy=False), dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _forecast_median_td(
    fc: Any, prediction_length: int, target_dim: int
) -> np.ndarray:
    pred = np.asarray(fc.quantile(0.5), dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if pred.shape == (prediction_length, target_dim):
        return pred
    if pred.shape == (target_dim, prediction_length):
        return pred.T
    raise ValueError(
        f"Unexpected forecast median shape {pred.shape}; "
        f"expected ({prediction_length}, {target_dim}) or transpose."
    )


def _forecast_samples_array(
    fc: Any, prediction_length: int, target_dim: int
) -> Optional[np.ndarray]:
    """
    Return ``(S, prediction_length, target_dim)`` from a GluonTS sample forecast, or ``None``.
    """
    samp = getattr(fc, "samples", None)
    if samp is None:
        return None
    a = np.asarray(samp, dtype=np.float64)
    if a.size == 0:
        return None
    pl = int(prediction_length)
    d = int(target_dim)
    if a.ndim == 2:
        s0, s1 = a.shape
        if s1 == pl and d == 1:
            return a.reshape(s0, pl, 1)
        if s0 == pl and d == 1:
            return a.T.reshape(-1, pl, 1)
    if a.ndim != 3:
        return None
    s0, s1, s2 = a.shape
    if s1 == pl and s2 == d:
        return a
    if s2 == pl and s1 == d:
        return np.transpose(a, (0, 2, 1))
    if s1 == pl and s2 != d and s2 == 1 and d == 1:
        return a.reshape(s0, pl, 1)
    return None


def _right_zoom_series_length(
    total_len: int,
    prediction_length: int,
    tail_steps: int,
    *,
    mode: str,
    forecast_context_steps: int,
) -> int:
    """How many tail time steps to draw in the right-hand zoom column."""
    T = int(total_len)
    pl = int(prediction_length)
    ts = max(3, int(tail_steps))
    if str(mode).lower() == "fixed_tail":
        return min(T, ts)
    ctx = max(3, int(forecast_context_steps))
    desired = pl + ctx
    cap = min(desired, ts, T)
    return max(3, cap)


def _tight_xy_lim_from_arrays(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    margin: float = 0.07,
) -> Optional[tuple[float, float, float, float]]:
    parts_x: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        xa = np.asarray(x, dtype=np.float64).ravel()
        ya = np.asarray(y, dtype=np.float64).ravel()
        if xa.size == 0:
            continue
        parts_x.append(xa)
        parts_y.append(ya)
    if not parts_x:
        return None
    allx = np.concatenate(parts_x)
    ally = np.concatenate(parts_y)
    minx, maxx = float(np.min(allx)), float(np.max(allx))
    miny, maxy = float(np.min(ally)), float(np.max(ally))
    dx = max(maxx - minx, 1e-3)
    dy = max(maxy - miny, 1e-3)
    m = float(margin)
    return (minx - m * dx, maxx + m * dx, miny - m * dy, maxy + m * dy)


def _targets_to_absolute_td(
    td: np.ndarray,
    raw0: np.ndarray,
    spatial_label: str,
) -> np.ndarray:
    """Invert ListDataset spatial transform to absolute state (same units as NPZ)."""
    r0 = np.asarray(raw0, dtype=np.float64).reshape(1, -1)
    td = np.asarray(td, dtype=np.float64)
    if spatial_label.startswith("delta_step_increment"):
        return np.cumsum(td, axis=0) + r0
    if spatial_label.startswith("delta_cumulative_from_first"):
        return td + r0
    return td.copy()


def _abs_forecast_trajectory_eval(
    meta: Dict[str, Any],
    npz_path: Path,
    forecasts: List[Any],
    tss: List[Any],
    test_traj_ids: List[int],
    min_len: int,
    target_dim: int,
    prediction_length: int,
    spatial_label: str,
    use_global_affine: bool,
    detail_panel_limit: int,
) -> Optional[dict]:
    """
    Reconstruct **absolute** (x,y,...) trajectories from NPZ + median forecast, compare on the
    **forecast window** only. Used for clear physical metrics and trajectory figures.

    Returns ``None`` if inputs are misaligned or ``target_dim < 2``.
    """
    if target_dim < 2 or not forecasts or len(test_traj_ids) != len(forecasts) or len(forecasts) != len(tss):
        return None
    pl = int(prediction_length)
    data_npz = np.load(npz_path, allow_pickle=True)
    trajs_root = data_npz["trajectories"]

    raw_xy_list: List[np.ndarray] = []
    panels: List[Dict[str, Any]] = []
    all_ex_fw: List[float] = []
    all_ey_fw: List[float] = []
    all_ev_fw: List[float] = []
    series_ade_abs_xy: List[float] = []
    first_gt_abs: Optional[np.ndarray] = None
    first_pr_abs: Optional[np.ndarray] = None

    for i, (fc, ts, tid) in enumerate(zip(forecasts, tss, test_traj_ids)):
        raw = np.asarray(trajs_root[int(tid)], dtype=np.float64)[:min_len]
        raw_xy_list.append(raw[:, :2].copy())
        ts_td = _ts_to_time_target(ts, target_dim)
        pred = _forecast_median_td(fc, pl, target_dim)
        if use_global_affine:
            ts_td = _denorm_metadata_zscore_channels(ts_td, meta)
            pred = _denorm_metadata_zscore_channels(pred, meta)
        fill = ts_td.copy()
        fill[-pl:] = pred
        abs_pred = _targets_to_absolute_td(fill, raw[0], spatial_label)
        abs_gt = raw.copy()
        if i == 0:
            first_gt_abs = abs_gt
            first_pr_abs = abs_pred
        ex = abs_pred[-pl:, 0] - abs_gt[-pl:, 0]
        ey = abs_pred[-pl:, 1] - abs_gt[-pl:, 1]
        epos = np.sqrt(ex**2 + ey**2)
        series_ade_abs_xy.append(float(np.mean(epos)))
        all_ex_fw.extend(ex.ravel().tolist())
        all_ey_fw.extend(ey.ravel().tolist())
        if target_dim >= 4:
            vx = abs_pred[-pl:, 2] - abs_gt[-pl:, 2]
            vy = abs_pred[-pl:, 3] - abs_gt[-pl:, 3]
            all_ev_fw.extend(np.sqrt(vx * vx + vy * vy).tolist())
        if len(panels) < detail_panel_limit:
            samples_td = _forecast_samples_array(fc, pl, target_dim)
            sample_stack_xy: Optional[np.ndarray] = None
            mean_xy: np.ndarray
            if samples_td is not None and samples_td.shape[0] > 0:
                paths: List[np.ndarray] = []
                for s in range(samples_td.shape[0]):
                    fill_s = ts_td.copy()
                    fill_s[-pl:] = samples_td[s]
                    abs_s = _targets_to_absolute_td(fill_s, raw[0], spatial_label)
                    paths.append(abs_s[:, :2].copy())
                sample_stack_xy = np.stack(paths, axis=0)
                mean_xy = np.mean(sample_stack_xy, axis=0)
            else:
                mean_xy = abs_pred[:, :2].copy()
            panels.append(
                {
                    "traj_id": int(tid),
                    "gt_xy": abs_gt[:, :2].copy(),
                    "median_xy": abs_pred[:, :2].copy(),
                    "mean_xy": mean_xy,
                    "sample_xy": sample_stack_xy,
                }
            )

    ex_arr = np.asarray(all_ex_fw, dtype=np.float64)
    ey_arr = np.asarray(all_ey_fw, dtype=np.float64)
    pooled_rmse_xy = float(np.sqrt(np.mean(ex_arr**2 + ey_arr**2)))
    pooled_rmse_vel: Optional[float] = None
    if all_ev_fw and target_dim >= 4:
        ev = np.asarray(all_ev_fw, dtype=np.float64)
        pooled_rmse_vel = float(np.sqrt(np.mean(ev**2)))

    sades = np.asarray(series_ade_abs_xy, dtype=np.float64)
    metrics = {
        "abs_forecast_pooled_rmse_xy_m": pooled_rmse_xy,
        "abs_forecast_mean_ade_xy_m": float(np.mean(sades)),
        "abs_forecast_median_ade_xy_m": float(np.median(sades)),
        "abs_forecast_pooled_rmse_vel_ms": pooled_rmse_vel,
    }
    return {
        "metrics": metrics,
        "raw_xy_list": raw_xy_list,
        "panels": panels,
        "first_gt_abs": first_gt_abs,
        "first_pr_abs": first_pr_abs,
        "pooled_rmse_xy_m": pooled_rmse_xy,
        "pooled_rmse_vel_ms": pooled_rmse_vel,
    }


def _save_trajectories_global_xy(
    out_dir: Path,
    raw_xy_list: List[np.ndarray],
    preset_name: str,
) -> Path:
    """All test trajectories in absolute (x,y), faint lines (Radar-style global view)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    for xy in raw_xy_list:
        ax.plot(xy[:, 0], xy[:, 1], "-", color="0.65", alpha=0.25, linewidth=0.9)
    ax.set_title(f"Test set ground-truth paths (absolute x,y) — {preset_name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    out = out_dir / "trajectories_global_test_xy.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def _slice_traj_panel_row(row: Dict[str, Any], start: int) -> Dict[str, Any]:
    """Slice time axis ``[start:]`` for zoomed views (``sample_xy`` is ``S×T×2``)."""
    sl = slice(int(start), None)
    samp = row.get("sample_xy")
    samp_sl = None
    if samp is not None and isinstance(samp, np.ndarray) and samp.ndim == 3:
        samp_sl = samp[:, sl, :].copy()
    return {
        "traj_id": row["traj_id"],
        "gt_xy": row["gt_xy"][sl].copy(),
        "median_xy": row["median_xy"][sl].copy(),
        "mean_xy": row["mean_xy"][sl].copy(),
        "sample_xy": samp_sl,
    }


def _draw_traj_layers_xy(
    ax: Any,
    row: Dict[str, Any],
    *,
    draw_samples: bool = True,
    end_markers: bool = True,
) -> None:
    """Draw sample fan (low alpha), mean, median, then GT on top (English-friendly legend)."""
    gt = row["gt_xy"]
    median = row["median_xy"]
    mean_xy = row["mean_xy"]
    samples = row.get("sample_xy") if draw_samples else None
    if samples is not None and isinstance(samples, np.ndarray) and samples.ndim == 3:
        s0 = samples.shape[0]
        for s in range(s0):
            ax.plot(
                samples[s, :, 0],
                samples[s, :, 1],
                color="0.55",
                linewidth=0.75,
                alpha=0.14,
                zorder=1,
                label="Sample paths" if s == 0 else "_nolegend_",
            )
    ax.plot(mean_xy[:, 0], mean_xy[:, 1], color="darkorange", linewidth=1.9, label="Mean", zorder=3)
    ax.plot(median[:, 0], median[:, 1], "r--", linewidth=1.45, alpha=0.95, label="Median", zorder=4)
    ax.plot(gt[:, 0], gt[:, 1], "g-", linewidth=2.1, label="GT", zorder=5)
    if end_markers and gt.shape[0] > 0:
        ax.scatter(gt[-1, 0], gt[-1, 1], c="green", s=40, zorder=6, marker="o", label="_nolegend_")
        ax.scatter(median[-1, 0], median[-1, 1], c="red", s=40, zorder=6, marker="x", label="_nolegend_")


def _save_trajectory_detail_panel(
    out_dir: Path,
    panels: List[Dict[str, Any]],
    preset_name: str,
) -> Path:
    """Per-series absolute (x,y): sample fan + mean + median vs GT (same units as NPZ)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * max(1, nrows)))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.set_axis_off()
    for k, row in enumerate(panels):
        ax = axes_flat[k]
        _draw_traj_layers_xy(ax, row, draw_samples=True, end_markers=True)
        tid = row["traj_id"]
        ax.set_title(f"Test traj id={tid} (absolute x,y, m)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        h, lab = ax.get_legend_handles_labels()
        uniq: Dict[str, Any] = {}
        for hi, li in zip(h, lab):
            if li and not li.startswith("_"):
                uniq[li] = hi
        if uniq:
            ax.legend(uniq.values(), uniq.keys(), loc="best", fontsize=8)
    fig.suptitle(f"GT vs TimeGrad — sample fan + mean + median — {preset_name}", fontsize=13)
    out = out_dir / "trajectory_detail_panel.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def _save_trajectory_full_and_tail_panel(
    out_dir: Path,
    panels: List[Dict[str, Any]],
    preset_name: str,
    *,
    tail_steps: int,
    prediction_length: int,
    right_panel_mode: str = "forecast_context",
    forecast_context_steps: int = 24,
) -> Path:
    """
    Left: full absolute (x,y). Right: **tight** zoom on the forecast end (default: last
    ``prediction_length + forecast_context_steps`` steps) with axis limits from drawn data.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    if n == 0:
        raise ValueError("panels empty")
    tail_steps = max(3, int(tail_steps))
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for k, row in enumerate(panels):
        ax_full, ax_tail = axes[k, 0], axes[k, 1]
        tid = row["traj_id"]
        _draw_traj_layers_xy(ax_full, row, draw_samples=True, end_markers=True)
        ax_full.set_title(f"id={tid} full path (absolute x,y, m)")
        ax_full.set_xlabel("x (m)")
        ax_full.set_ylabel("y (m)")
        ax_full.axis("equal")
        ax_full.grid(True, alpha=0.3)
        h0, l0 = ax_full.get_legend_handles_labels()
        u0 = {li: hi for hi, li in zip(h0, l0) if li and not li.startswith("_")}
        if u0:
            ax_full.legend(u0.values(), u0.keys(), loc="best", fontsize=8)

        T = int(row["gt_xy"].shape[0])
        m = _right_zoom_series_length(
            T,
            int(prediction_length),
            tail_steps,
            mode=right_panel_mode,
            forecast_context_steps=int(forecast_context_steps),
        )
        start = max(0, T - m)
        zoom_row = _slice_traj_panel_row(row, start)
        _draw_traj_layers_xy(ax_tail, zoom_row, draw_samples=True, end_markers=True)
        mode_lbl = "forecast+context" if str(right_panel_mode).lower() != "fixed_tail" else "fixed tail"
        ax_tail.set_title(f"id={tid} zoom ({mode_lbl}, last {m} steps)")
        ax_tail.set_xlabel("x (m)")
        ax_tail.set_ylabel("y (m)")
        ax_tail.grid(True, alpha=0.3)
        h1, l1 = ax_tail.get_legend_handles_labels()
        u1 = {li: hi for hi, li in zip(h1, l1) if li and not li.startswith("_")}
        if u1:
            ax_tail.legend(u1.values(), u1.keys(), loc="best", fontsize=8)

        xs: List[np.ndarray] = [
            zoom_row["gt_xy"][:, 0],
            zoom_row["median_xy"][:, 0],
            zoom_row["mean_xy"][:, 0],
        ]
        ys: List[np.ndarray] = [
            zoom_row["gt_xy"][:, 1],
            zoom_row["median_xy"][:, 1],
            zoom_row["mean_xy"][:, 1],
        ]
        zs = zoom_row.get("sample_xy")
        if zs is not None and isinstance(zs, np.ndarray) and zs.ndim == 3:
            for s in range(zs.shape[0]):
                xs.append(zs[s, :, 0])
                ys.append(zs[s, :, 1])
        lim = _tight_xy_lim_from_arrays(xs, ys, margin=0.08)
        if lim is not None:
            xmin, xmax, ymin, ymax = lim
            ax_tail.set_xlim(xmin, xmax)
            ax_tail.set_ylim(ymin, ymax)
        ax_tail.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Full path vs end zoom — GT / mean / median / samples — {preset_name}",
        fontsize=13,
    )
    out = out_dir / "trajectory_full_and_tail_panel.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def _save_error_comparison_timegrad(
    out_dir: Path,
    gt_abs: np.ndarray,
    pred_abs: np.ndarray,
    pooled_rmse_pos_m: float,
    pooled_rmse_vel_ms: Optional[float],
    preset_name: str,
) -> Path:
    """
    2×2 layout following ``Radar/examples/particle_filter_example.py``:
    trajectory | planar pos error vs time | vel error vs time | RMSE bars.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = gt_abs.shape[0]
    time_steps = np.arange(T)
    pos_err = np.sqrt((pred_abs[:, 0] - gt_abs[:, 0]) ** 2 + (pred_abs[:, 1] - gt_abs[:, 1]) ** 2)
    rmse_pos = float(np.sqrt(np.mean(pos_err**2)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1 = axes[0, 0]
    ax1.plot(gt_abs[:, 0], gt_abs[:, 1], "g-", linewidth=2, label="Ground truth")
    ax1.plot(pred_abs[:, 0], pred_abs[:, 1], "r--", linewidth=1.5, label="TimeGrad median (recon)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Trajectory comparison (first test series)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    ax2 = axes[0, 1]
    ax2.plot(time_steps, pos_err, "darkred", linewidth=1.5, alpha=0.85, label="|pos err| per step")
    ax2.axhline(y=rmse_pos, color="r", linestyle="--", alpha=0.5, label=f"RMSE_pos={rmse_pos:.2f} m")
    ax2.set_xlabel("Time step (within cropped window)")
    ax2.set_ylabel("Planar position error (m)")
    ax2.set_title("Position error vs time (GT vs recon)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if gt_abs.shape[1] >= 4:
        vel_err = np.sqrt(np.sum((pred_abs[:, 2:4] - gt_abs[:, 2:4]) ** 2, axis=1))
        rmse_vel = float(np.sqrt(np.mean(vel_err**2)))
        ax3.plot(time_steps, vel_err, "purple", linewidth=1.5, alpha=0.85, label="|vel err| per step")
        ax3.axhline(
            y=rmse_vel,
            color="purple",
            linestyle="--",
            alpha=0.5,
            label=f"RMSE_vel={rmse_vel:.3f} m/s",
        )
        ax3.set_xlabel("Time step")
        ax3.set_ylabel("Velocity error (m/s)")
        ax3.set_title("Velocity error vs time")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "target_dim < 4: no velocity", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()
        rmse_vel = None

    ax4 = axes[1, 1]
    if pooled_rmse_vel_ms is not None and np.isfinite(pooled_rmse_vel_ms):
        categories = ["Position (m)", "Velocity (m/s)"]
        vals = [pooled_rmse_pos_m, pooled_rmse_vel_ms]
    else:
        categories = ["Position (m)"]
        vals = [pooled_rmse_pos_m]
    x = np.arange(len(categories))
    bars = ax4.bar(x, vals, width=0.5, color="steelblue", alpha=0.85)
    for bar, val in zip(bars, vals):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * (max(vals) + 1e-6),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_ylabel("RMSE (forecast window, pooled over test)")
    ax4.set_title("RMSE — median forecast vs GT")
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Error comparison (TimeGrad) — {preset_name}", fontsize=13)
    out = out_dir / "error_comparison_timegrad.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def run_single(
    dataset_dir: Path,
    meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    freq = str(meta["freq"])
    prediction_length = int(meta["prediction_length"])
    context_length = int(meta["context_length"])
    target_dim = int(meta["target_dim"])
    min_len = _required_past_length(freq, context_length) + prediction_length + 2

    apply_loader_md_z = bool(getattr(args, "dataset_target_metadata_zscore", False))
    use_global_affine = bool(meta.get("dataset_target_global_zscore", False)) or apply_loader_md_z

    train_ds, _ = _build_list_dataset(
        meta,
        list(meta["train_indices"]),
        min_len,
        apply_metadata_zscore=apply_loader_md_z,
    )
    test_ds, test_traj_ids = _build_list_dataset(
        meta,
        list(meta["test_indices"]),
        min_len,
        apply_metadata_zscore=apply_loader_md_z,
    )

    diff_steps = int(meta.get("diff_steps", 100))
    beta_end = float(meta.get("beta_end", 0.1))
    scheduler = DDPMScheduler(
        num_train_timesteps=diff_steps,
        beta_schedule="linear",
        beta_end=beta_end,
        prediction_type="epsilon",
    )

    if getattr(args, "timegrad_scaling", None) is not None:
        scaling = str(args.timegrad_scaling)
    else:
        scaling = "mean" if meta.get("timegrad_scaling", True) else "none"
    spatial_label, ch01_step_increments = _spatial_transform_label(meta)
    hidden_size = int(args.hidden_size or meta.get("num_cells", 40))
    num_layers = int(args.num_layers or meta.get("num_layers", 2))

    if args.encoder_type == "transformer" and hidden_size % int(args.transformer_nhead) != 0:
        raise ValueError(
            f"hidden_size {hidden_size} must be divisible by transformer_nhead "
            f"{args.transformer_nhead}"
        )

    sys.path.insert(0, str(REPO_ROOT))
    from pts.model.time_grad.estimator import TimeGradEstimator
    from pts.model.time_grad.lightning_module import TimeGradLightningModule

    # PyTorch 2.6+ defaults weights_only=True in torch.load; checkpoints embed the
    # diffusers scheduler, which safe unpickling rejects without allowlisting.
    _orig_load_ckpt = TimeGradLightningModule.load_from_checkpoint

    def _load_ckpt_weights_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load_ckpt(*args, **kwargs)

    TimeGradLightningModule.load_from_checkpoint = _load_ckpt_weights_compat  # type: ignore[method-assign]

    try:
        accum = max(1, int(getattr(args, "accumulate_grad_batches", 1)))
        trainer_kwargs: Dict[str, Any] = {
            "max_epochs": int(args.max_epochs),
            "gradient_clip_val": 10.0,
            "accelerator": args.accelerator,
            "devices": 1,
            "accumulate_grad_batches": accum,
        }
        if args.accelerator == "gpu":
            trainer_kwargs["devices"] = min(1, torch.cuda.device_count() or 1)

        lags_seq = None
        if args.lags_preset == "metadata":
            lags_seq = meta.get("lags_seq")
            if lags_seq is not None:
                lags_seq = [int(x) for x in lags_seq]
        elif args.lags_preset == "short":
            lags_seq = list(range(1, min(9, context_length + 1)))

        if apply_loader_md_z and scaling not in ("none",):
            print(
                "WARN: --dataset-target-metadata-zscore with GluonTS scaling!=none stacks two "
                "normalizations; prefer e.g. GPU A: --timegrad-scaling mean alone; "
                "GPU B: --timegrad-scaling none + --dataset-target-metadata-zscore.",
                flush=True,
            )
        print(
            f"Train: batch={int(args.batch_size)}×accum={accum} "
            f"(eff {int(args.batch_size) * accum})  lr={float(args.lr)}  "
            f"epochs={int(args.max_epochs)}  batches/epoch={int(args.num_batches_per_epoch)}  "
            f"scaling={scaling}  spatial={spatial_label}  "
            f"metadata_zscore_loader={'on' if apply_loader_md_z else 'off'}",
            flush=True,
        )

        estimator = TimeGradEstimator(
            freq=freq,
            prediction_length=prediction_length,
            input_size=target_dim,
            scheduler=scheduler,
            context_length=context_length,
            num_layers=num_layers,
            hidden_size=hidden_size,
            lr=float(args.lr),
            dropout_rate=float(args.dropout_rate),
            num_feat_dynamic_real=0,
            num_feat_static_cat=0,
            num_feat_static_real=0,
            scaling=scaling,
            num_parallel_samples=int(args.num_parallel_samples),
            num_inference_steps=int(args.num_inference_steps),
            encoder_type=str(args.encoder_type),
            transformer_nhead=int(args.transformer_nhead),
            transformer_dim_feedforward=(
                int(args.transformer_dim_feedforward)
                if args.transformer_dim_feedforward is not None
                else None
            ),
            batch_size=int(args.batch_size),
            num_batches_per_epoch=int(args.num_batches_per_epoch),
            lags_seq=lags_seq,
            trainer_kwargs=trainer_kwargs,
        )

        t0 = time.perf_counter()
        predictor = estimator.train(
            training_data=train_ds,
            shuffle_buffer_length=max(1024, int(args.batch_size) * 8),
        )
        train_seconds = time.perf_counter() - t0

        num_samples = min(int(args.eval_num_samples), int(args.num_parallel_samples))
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,
            predictor=predictor,
            num_samples=num_samples,
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)

        evaluator = Evaluator(quantiles=(0.1, 0.5, 0.9), num_workers=0)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(tss))

        mean = meta.get("normalization_mean")
        std = meta.get("normalization_std")
        # NPZ trajectories are usually **raw** units; global mean/std are for
        # diagnostics or z-scored exports. Only apply inverse z-score when set.
        pairs = _collect_aligned_pred_obs(
            forecasts, tss, prediction_length, mean, std, use_global_affine=use_global_affine
        )
        denorm = None
        if meta.get("denorm_metrics") and pairs:
            denorm = float(
                np.mean([float(np.sqrt(np.mean((p - o) ** 2))) for p, o in pairs])
            )

        out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "timegrad_autoruns"
        out_dir = out_dir / f"{int(time.time())}_{args.encoder_type}"
        out_dir.mkdir(parents=True, exist_ok=True)

        position_metrics: Dict[str, float] = {}
        position_eval_plot: Optional[str] = None
        trajectory_plot_paths: Dict[str, str] = {}
        if (
            target_dim >= 2
            and pairs
            and not getattr(args, "no_eval_plots", False)
        ):
            pos_stats, pos_arrays = _position_velocity_eval(pairs, target_dim)
            position_metrics = pos_stats
            plot_path = _save_position_eval_figure(
                out_dir,
                pos_stats,
                pos_arrays,
                dataset_dir.name,
                ch01_step_increments=ch01_step_increments,
            )
            position_eval_plot = str(plot_path)

        traj_eval: Optional[dict] = None
        if target_dim >= 2 and forecasts and len(test_traj_ids) == len(forecasts) == len(tss):
            npz_path = Path(meta["input_npz"])
            detail_n = max(0, int(getattr(args, "eval_trajectory_detail_panels", 4)))
            traj_eval = _abs_forecast_trajectory_eval(
                meta,
                npz_path,
                forecasts,
                tss,
                test_traj_ids,
                min_len,
                target_dim,
                prediction_length,
                spatial_label,
                use_global_affine,
                detail_n,
            )
            if traj_eval is not None and not getattr(args, "no_eval_plots", False):
                raw_xy_list = traj_eval["raw_xy_list"]
                panels = traj_eval["panels"]
                first_gt_abs = traj_eval["first_gt_abs"]
                first_pr_abs = traj_eval["first_pr_abs"]
                pooled_rmse_pos_m = float(traj_eval["pooled_rmse_xy_m"])
                pooled_rmse_vel_ms = traj_eval["pooled_rmse_vel_ms"]
                trajectory_plot_paths["global_xy"] = str(
                    _save_trajectories_global_xy(out_dir, raw_xy_list, dataset_dir.name)
                )
                if panels:
                    trajectory_plot_paths["detail_panel"] = str(
                        _save_trajectory_detail_panel(out_dir, panels, dataset_dir.name)
                    )
                    tail_n = max(3, int(args.eval_trajectory_tail_steps))
                    trajectory_plot_paths["full_and_tail"] = str(
                        _save_trajectory_full_and_tail_panel(
                            out_dir,
                            panels,
                            dataset_dir.name,
                            tail_steps=tail_n,
                            prediction_length=prediction_length,
                            right_panel_mode=str(
                                getattr(args, "eval_trajectory_right_panel", "forecast_context")
                            ),
                            forecast_context_steps=int(
                                getattr(args, "eval_trajectory_forecast_context_steps", 24)
                            ),
                        )
                    )
                if first_gt_abs is not None and first_pr_abs is not None:
                    trajectory_plot_paths["error_comparison"] = str(
                        _save_error_comparison_timegrad(
                            out_dir,
                            first_gt_abs,
                            first_pr_abs,
                            pooled_rmse_pos_m,
                            pooled_rmse_vel_ms,
                            dataset_dir.name,
                        )
                    )

        agg_json = _metrics_for_json(agg_metrics)
        terminal_metrics: Dict[str, Any] = {
            "MASE": agg_json.get("MASE"),
            "train_seconds": float(train_seconds),
        }
        if traj_eval is not None:
            terminal_metrics.update(traj_eval["metrics"])

        metric_glossary_cn = {
            "MASE": "相对朴素基线的平均绝对缩放误差；<1 通常优于基线，可跨长度比较。",
            "train_seconds": "本轮训练墙钟时间（秒）。",
            "abs_forecast_pooled_rmse_xy_m": (
                "全部测试序列、预测窗内，由 NPZ 绝对坐标重建后的平面 (x,y) 误差均方根（米）；"
                "适合作为同一数据集上的主参考指标。"
            ),
            "abs_forecast_mean_ade_xy_m": (
                "每条测试序列在预测窗上对绝对 (x,y) 的平均 L2 误差（米），再对所有序列取算术平均。"
            ),
            "abs_forecast_median_ade_xy_m": "同上，但对序列取中位数，抗离群。",
            "abs_forecast_pooled_rmse_vel_ms": (
                "预测窗内水平速度矢量模的均方根误差（m/s）；无速度通道或未计算时为 null。"
            ),
        }

        summary = {
            "dataset_dir": str(dataset_dir),
            "dataset_spatial_transform": spatial_label,
            "delta_encoding_resolved": _resolved_delta_encoding(meta)
            if str(meta.get("state_repr", "absolute")).lower() == "delta"
            and str(meta.get("position_origin", "none")).lower() == "first"
            else None,
            "timegrad_scaling_effective": scaling,
            "timegrad_scaling_cli_override": getattr(args, "timegrad_scaling", None),
            "position_ch01_semantics": (
                "per_step_delta_m" if ch01_step_increments else "absolute_xy_m"
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "encoder_type": args.encoder_type,
            "transformer_nhead": int(args.transformer_nhead),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "lags_preset": args.lags_preset,
            "max_epochs": int(args.max_epochs),
            "batch_size": int(args.batch_size),
            "accumulate_grad_batches": int(getattr(args, "accumulate_grad_batches", 1)),
            "effective_batch_size": int(args.batch_size)
            * int(getattr(args, "accumulate_grad_batches", 1)),
            "lr": float(args.lr),
            "num_batches_per_epoch": int(args.num_batches_per_epoch),
            "train_seconds": train_seconds,
            "num_train_series": len(train_ds),
            "num_test_series": len(test_ds),
            "min_length_required": min_len,
            "agg_metrics": _metrics_for_json(agg_metrics),
            "denorm_median_rmse": denorm,
            "dataset_target_global_zscore": use_global_affine,
            "dataset_target_metadata_zscore_loader": apply_loader_md_z,
            "position_metrics": position_metrics,
            "position_eval_plot": position_eval_plot,
            "trajectory_plot_paths": trajectory_plot_paths or None,
            "artifact_dir": str(out_dir),
            "terminal_metrics": terminal_metrics,
            "metric_glossary_cn": metric_glossary_cn,
        }
        with (out_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        item_metrics.to_csv(out_dir / "per_series_metrics.csv", index=False)

        return summary
    finally:
        TimeGradLightningModule.load_from_checkpoint = _orig_load_ckpt  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser(description="TimeGrad train + eval on preprocess_metadata datasets.")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Subfolder name under dataset/, e.g. radar_viz_debug",
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="*",
        default=["radar_smoke_gpu", "radar_viz_debug", "exp_motion_vrw_gpu1"],
        help="Run several presets in one invocation (ignored if --preset is set).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="Root directory containing preset subfolders.",
    )
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument(
        "--lags-preset",
        choices=["freq", "short", "metadata"],
        default="metadata",
        help="Lag set: GluonTS defaults for freq, short 1..8, or lags_seq from preprocess_metadata.json when present.",
    )
    parser.add_argument("--encoder-type", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument(
        "--timegrad-scaling",
        type=str,
        choices=["mean", "none", "std"],
        default=None,
        metavar="S",
        help="Override preprocess_metadata timegrad_scaling for this run (default: use JSON).",
    )
    parser.add_argument(
        "--dataset-target-metadata-zscore",
        action="store_true",
        help=(
            "After spatial transforms, apply per-channel (x,y,vx,vy) "
            "(target - normalization_mean) / normalization_std from JSON, then denorm at eval. "
            "For a clean A/B vs GluonTS scaler, pair with --timegrad-scaling none; do not combine "
            "with mean/std scaling unless you intend stacked normalization."
        ),
    )
    parser.add_argument("--accelerator", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Per-step micro-batch for GluonTS/Lightning (default 256 on large GPUs; reduce if OOM).",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        metavar="N",
        help="Lightning accumulate_grad_batches: effective batch = batch_size * N (more GPU work before optimizer).",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-batches-per-epoch", type=int, default=None)
    parser.add_argument("--num-parallel-samples", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--eval-num-samples", type=int, default=50)
    parser.add_argument(
        "--no-eval-plots",
        action="store_true",
        help="Skip saving evaluation figures (position + trajectories + error comparison).",
    )
    parser.add_argument(
        "--verbose-eval-metrics",
        action="store_true",
        help="Print full GluonTS / denorm / increment-space metrics to the terminal (default: short summary only).",
    )
    parser.add_argument(
        "--eval-trajectory-detail-panels",
        type=int,
        default=4,
        metavar="N",
        help="Number of test series in trajectory_detail_panel.png (default 4 in 2×2).",
    )
    parser.add_argument(
        "--eval-trajectory-tail-steps",
        type=int,
        default=80,
        metavar="T",
        help=(
            "Upper cap / tail length when --eval-trajectory-right-panel=fixed_tail. "
            "For default forecast_context mode, the right column uses prediction_length + "
            "--eval-trajectory-forecast-context-steps (this flag is only a bound in that mode)."
        ),
    )
    parser.add_argument(
        "--eval-trajectory-right-panel",
        type=str,
        choices=["forecast_context", "fixed_tail"],
        default="forecast_context",
        help=(
            "Right column of trajectory_full_and_tail_panel: "
            "'forecast_context' = last prediction_length + context steps (fine zoom near the forecast); "
            "'fixed_tail' = last --eval-trajectory-tail-steps steps (legacy)."
        ),
    )
    parser.add_argument(
        "--eval-trajectory-forecast-context-steps",
        type=int,
        default=24,
        metavar="C",
        help="Extra past steps before the forecast window in the right-hand zoom (forecast_context mode).",
    )

    args = parser.parse_args()

    # GluonTS 0.16 + PyTorch 2.x: repeat_along_dim uses list indexing; harmless but very noisy.
    warnings.filterwarnings(
        "ignore",
        message=r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        category=UserWarning,
    )

    if args.accelerator == "auto":
        args.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if args.accelerator == "gpu" and torch.cuda.is_available():
        # Better Tensor Core throughput on Ada/Blackwell (Lightning prints a hint otherwise).
        torch.set_float32_matmul_precision("medium")

    if args.mode == "smoke":
        args.max_epochs = 2 if args.max_epochs is None else args.max_epochs
        args.num_batches_per_epoch = 25 if args.num_batches_per_epoch is None else args.num_batches_per_epoch
        args.batch_size = min(int(args.batch_size), 16)
        args.accumulate_grad_batches = 1
        args.num_inference_steps = min(int(args.num_inference_steps), 25)
        args.eval_num_samples = min(int(args.eval_num_samples), 20)
    else:
        args.max_epochs = 15 if args.max_epochs is None else args.max_epochs
        args.num_batches_per_epoch = 120 if args.num_batches_per_epoch is None else args.num_batches_per_epoch

    presets = [args.preset] if args.preset else list(args.presets)
    rows = []
    for name in presets:
        dataset_dir = (args.dataset_root / name).resolve()
        print(f"\n=== {name} ({dataset_dir}) ===", flush=True)
        try:
            meta = _load_metadata(dataset_dir)
            summary = run_single(dataset_dir, meta, args)
            rows.append(summary)
            verbose = bool(getattr(args, "verbose_eval_metrics", False))

            def _fmt_metric(v: Any) -> str:
                if v is None:
                    return "n/a"
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return "n/a"
                if not np.isfinite(fv):
                    return "n/a"
                return f"{fv:.4g}"

            if verbose:
                m = summary["agg_metrics"]
                line = (
                    f"MSE={m.get('MSE', float('nan')):.6g}  "
                    f"MASE={m.get('MASE', float('nan')):.6g}  "
                    f"train_s={summary['train_seconds']:.1f}"
                )
                if summary.get("denorm_median_rmse") is not None:
                    tag = (
                        "median_RMSE(z->phys)"
                        if summary.get("dataset_target_global_zscore")
                        else "median_RMSE(target_units)"
                    )
                    line += f"  {tag}={summary['denorm_median_rmse']:.6g}"
                pm = summary.get("position_metrics") or {}
                if pm:
                    dtag = (
                        " [ch0-1=Δstep]"
                        if summary.get("position_ch01_semantics") == "per_step_delta_m"
                        else ""
                    )
                    line += (
                        f"  ADE_xy(median)={pm.get('ade_xy_median_m', float('nan')):.2f}m"
                        f"  ADE_xy(mean)={pm.get('ade_xy_mean_m', float('nan')):.2f}m"
                        f"  RMSE_x={pm.get('rmse_x_m', float('nan')):.2f}m"
                        f"  RMSE_y={pm.get('rmse_y_m', float('nan')):.2f}m"
                        f"{dtag}"
                    )
                    if "rmse_vel_plane_ms" in pm:
                        line += f"  RMSE_vxy={pm['rmse_vel_plane_ms']:.3f}m/s"
                print(line, flush=True)
                if summary.get("position_eval_plot"):
                    print(f"  position plot: {summary['position_eval_plot']}", flush=True)
                tp = summary.get("trajectory_plot_paths") or {}
                for k in ("global_xy", "detail_panel", "full_and_tail", "error_comparison"):
                    p = tp.get(k)
                    if p:
                        print(f"  trajectory plot ({k}): {p}", flush=True)
            else:
                tm = summary.get("terminal_metrics") or {}
                parts = [
                    f"MASE={_fmt_metric(tm.get('MASE'))}",
                    (
                        "绝对平面·预测窗·池化RMSE_xy="
                        f"{_fmt_metric(tm.get('abs_forecast_pooled_rmse_xy_m'))}m"
                    ),
                    (
                        "绝对平面·预测窗·平均ADE_xy="
                        f"{_fmt_metric(tm.get('abs_forecast_mean_ade_xy_m'))}m"
                    ),
                    f"train_s={summary['train_seconds']:.0f}s",
                ]
                rv = tm.get("abs_forecast_pooled_rmse_vel_ms")
                if rv is not None and np.isfinite(float(rv)):
                    parts.append(f"RMSE_vxy={float(rv):.3f}m/s")
                print("  评估  " + "  ".join(parts), flush=True)
                print(f"  → {summary.get('artifact_dir', '')}", flush=True)
        except Exception as e:
            print(f"FAILED {name}: {e}", flush=True)
            rows.append({"dataset_dir": str(dataset_dir), "error": str(e)})

    out = REPO_ROOT / "dataset" / "_timegrad_last_run_summary.json"
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote combined summary to {out}", flush=True)


if __name__ == "__main__":
    main()
