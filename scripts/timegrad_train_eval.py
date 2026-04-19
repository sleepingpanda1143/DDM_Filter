#!/usr/bin/env python3
"""
Train and evaluate GluonTS-style TimeGrad (diffusers + pts TimeGradEstimator) on
pre-built dataset folders that contain ``preprocess_metadata.json`` (radar / motion runs).

Example:
  ./scripts/run_timegrad.sh smoke
  ./scripts/run_timegrad.sh train --preset radar_viz_debug --max-epochs 20
  TIMEGRAD_CUDA_DEVICE=0 ./scripts/run_timegrad.sh train ...   # 仅用物理 GPU 0

After ``train`` / ``smoke``, for ``target_dim >= 2`` saves ``position_step_eval.png`` (planar ADE/FDE
histograms + velocity + numeric summary) next to ``summary.json``. Use ``--no-eval-plots`` to skip.
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


def _build_list_dataset(
    meta: Dict[str, Any],
    indices: List[int],
    min_length: int,
) -> ListDataset:
    npz_path = Path(meta["input_npz"])
    if not npz_path.is_file():
        raise FileNotFoundError(f"input_npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "trajectories" not in data.files:
        raise KeyError(f"{npz_path} has no 'trajectories' array ({data.files})")

    trajs = data["trajectories"]
    freq = str(meta["freq"])
    entries = []
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
    if not entries:
        raise RuntimeError(
            f"No series left after min_length={min_length} filter "
            f"(check trajectories vs context/lags for freq={freq})."
        )
    return ListDataset(entries, freq=freq, one_dim_target=(int(meta["target_dim"]) == 1))


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
) -> Path:
    """2×2 figure similar in spirit to Radar ``error_comparison`` / PF example plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"TimeGrad median forecast — planar position & velocity ({preset_name})",
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
    ax0.set_xlabel("ADE_xy per series (m)")
    ax0.set_ylabel("Count")
    ax0.set_title("Distribution of mean planar position error over forecast horizon")
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
    ax1.set_xlabel("FDE_xy per series (m)")
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
    lines = [
        "Pooled RMSE (all test points, horizon):",
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

    train_ds = _build_list_dataset(meta, list(meta["train_indices"]), min_len)
    test_ds = _build_list_dataset(meta, list(meta["test_indices"]), min_len)

    diff_steps = int(meta.get("diff_steps", 100))
    beta_end = float(meta.get("beta_end", 0.1))
    scheduler = DDPMScheduler(
        num_train_timesteps=diff_steps,
        beta_schedule="linear",
        beta_end=beta_end,
        prediction_type="epsilon",
    )

    scaling = "mean" if meta.get("timegrad_scaling", True) else "none"
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
        trainer_kwargs: Dict[str, Any] = {
            "max_epochs": int(args.max_epochs),
            "gradient_clip_val": 10.0,
            "accelerator": args.accelerator,
            "devices": 1,
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
            shuffle_buffer_length=max(512, int(args.batch_size) * 8),
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
        use_affine = bool(meta.get("dataset_target_global_zscore", False))
        pairs = _collect_aligned_pred_obs(
            forecasts, tss, prediction_length, mean, std, use_global_affine=use_affine
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
        if (
            target_dim >= 2
            and pairs
            and not getattr(args, "no_eval_plots", False)
        ):
            pos_stats, pos_arrays = _position_velocity_eval(pairs, target_dim)
            position_metrics = pos_stats
            plot_path = _save_position_eval_figure(
                out_dir, pos_stats, pos_arrays, dataset_dir.name
            )
            position_eval_plot = str(plot_path)

        summary = {
            "dataset_dir": str(dataset_dir),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "encoder_type": args.encoder_type,
            "transformer_nhead": int(args.transformer_nhead),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "lags_preset": args.lags_preset,
            "max_epochs": int(args.max_epochs),
            "batch_size": int(args.batch_size),
            "num_batches_per_epoch": int(args.num_batches_per_epoch),
            "train_seconds": train_seconds,
            "num_train_series": len(train_ds),
            "num_test_series": len(test_ds),
            "min_length_required": min_len,
            "agg_metrics": _metrics_for_json(agg_metrics),
            "denorm_median_rmse": denorm,
            "dataset_target_global_zscore": use_affine,
            "position_metrics": position_metrics,
            "position_eval_plot": position_eval_plot,
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
    parser.add_argument("--accelerator", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size (default 128 for better GPU util on large GPUs; use smaller if OOM).",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-batches-per-epoch", type=int, default=None)
    parser.add_argument("--num-parallel-samples", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--eval-num-samples", type=int, default=50)
    parser.add_argument(
        "--no-eval-plots",
        action="store_true",
        help="Skip saving position_step_eval.png (matplotlib) after training.",
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
        args.num_inference_steps = min(int(args.num_inference_steps), 25)
        args.eval_num_samples = min(int(args.eval_num_samples), 20)
    else:
        args.max_epochs = 15 if args.max_epochs is None else args.max_epochs
        args.num_batches_per_epoch = 100 if args.num_batches_per_epoch is None else args.num_batches_per_epoch

    presets = [args.preset] if args.preset else list(args.presets)
    rows = []
    for name in presets:
        dataset_dir = (args.dataset_root / name).resolve()
        print(f"\n=== {name} ({dataset_dir}) ===", flush=True)
        try:
            meta = _load_metadata(dataset_dir)
            summary = run_single(dataset_dir, meta, args)
            rows.append(summary)
            m = summary["agg_metrics"]
            line = (
                f"MSE={m.get('MSE', float('nan')):.6g}  "
                f"MASE={m.get('MASE', float('nan')):.6g}  "
                f"train_s={summary['train_seconds']:.1f}"
            )
            if summary.get("denorm_median_rmse") is not None:
                tag = "median_RMSE(z->phys)" if summary.get("dataset_target_global_zscore") else "median_RMSE(target_units)"
                line += f"  {tag}={summary['denorm_median_rmse']:.6g}"
            pm = summary.get("position_metrics") or {}
            if pm:
                line += (
                    f"  ADE_xy(median)={pm.get('ade_xy_median_m', float('nan')):.2f}m"
                    f"  ADE_xy(mean)={pm.get('ade_xy_mean_m', float('nan')):.2f}m"
                    f"  RMSE_x={pm.get('rmse_x_m', float('nan')):.2f}m"
                    f"  RMSE_y={pm.get('rmse_y_m', float('nan')):.2f}m"
                )
                if "rmse_vel_plane_ms" in pm:
                    line += f"  RMSE_vxy={pm['rmse_vel_plane_ms']:.3f}m/s"
            print(line, flush=True)
            if summary.get("position_eval_plot"):
                print(f"  position plot: {summary['position_eval_plot']}", flush=True)
        except Exception as e:
            print(f"FAILED {name}: {e}", flush=True)
            rows.append({"dataset_dir": str(dataset_dir), "error": str(e)})

    out = REPO_ROOT / "dataset" / "_timegrad_last_run_summary.json"
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote combined summary to {out}", flush=True)


if __name__ == "__main__":
    main()
