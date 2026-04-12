#!/usr/bin/env python3
"""
Visualize a completed radar_timegrad_experiment run: GT paths, CV baseline errors,
and (if present) TimeGrad forecast sample clouds in x–y.

Does not require loading the serialized predictor (often breaks across versions).

Usage:
  PYTHONPATH=/path/to/DDM_Timeseries_Forecast \\
    python visualize_radar_timegrad_run.py --run-dir /path/to/radar_compare_runs_gpu0/02_long_train_double_scale

Re-run training with --save-forecast-samples to create forecast_samples.npz for model panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


def denormalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return array * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def cv_predict_one_step(prev_state: np.ndarray) -> np.ndarray:
    out = prev_state.copy()
    out[0] = prev_state[0] + prev_state[2]
    out[1] = prev_state[1] + prev_state[3]
    out[2] = prev_state[2]
    out[3] = prev_state[3]
    return out


def series_forecast_bundle(
    base: np.ndarray,
    tz: np.ndarray,
    samples: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    context_length: int,
    prediction_length: int,
    state_repr: str,
    denorm: bool,
) -> dict:
    """Physical-space context, truth, sample trajectories, mean, and CV one-step prediction."""
    base = np.asarray(base, dtype=np.float32)
    tz = np.asarray(tz, dtype=np.float32)
    samples = np.asarray(samples, dtype=np.float32)
    ctx_phys = base[-context_length - prediction_length : -prediction_length, :]
    true_phys = base[-prediction_length:, :]

    if denorm:
        samp_p = denormalize(samples, mean, std)
    else:
        samp_p = samples

    if state_repr == "delta":
        anchor = base[-prediction_length - 1].copy()
        true_p = true_phys
        samp_abs = []
        for s in range(samp_p.shape[0]):
            running = anchor.copy()
            row = []
            for step in range(prediction_length):
                running = running + samp_p[s, step]
                row.append(running.copy())
            samp_abs.append(np.stack(row, axis=0))
        samp_abs = np.stack(samp_abs, axis=0)
        mean_abs = samp_abs.mean(axis=0)
        ctx_p = ctx_phys
    else:
        samp_abs = samp_p
        mean_abs = samp_p.mean(axis=0)
        if denorm:
            ctx_p = ctx_phys
            true_p = true_phys
        else:
            ctx_p = tz[-context_length - prediction_length : -prediction_length, :]
            true_p = tz[-prediction_length:, :]

    cv_anchor = base[-prediction_length - 1].copy()
    cv_pred = cv_predict_one_step(cv_anchor)
    return {
        "ctx_p": ctx_p,
        "true_p": true_p,
        "mean_abs": mean_abs,
        "samp_abs": samp_abs,
        "cv_pred": cv_pred,
    }


def zoom_limits_around_forecast(
    bundle: dict,
    *,
    context_tail: int,
    margin_frac: float,
) -> tuple[float, float, float, float]:
    """xmin, xmax, ymin, ymax in physical coordinates."""
    ctx_p = bundle["ctx_p"]
    true_p = bundle["true_p"]
    mean_abs = bundle["mean_abs"]
    samp_abs = bundle["samp_abs"]
    cv_pred = bundle["cv_pred"]

    pts = [true_p[-1, :2], mean_abs[-1, :2], cv_pred[:2]]
    pts.extend(samp_abs[:, -1, :2].tolist())
    tail = max(1, min(context_tail, len(ctx_p)))
    pts.extend(ctx_p[-tail:, :2].tolist())
    arr = np.asarray(pts, dtype=np.float64)
    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    span = float(max(xmax - xmin, ymax - ymin, 1e-9))
    pad = span * margin_frac
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Output dir of radar_timegrad_experiment.")
    parser.add_argument(
        "--max-spaghetti",
        type=int,
        default=60,
        help="Number of test trajectories to draw as GT spaghetti.",
    )
    parser.add_argument(
        "--max-panels",
        type=int,
        default=8,
        help="Number of per-series panels when forecast_samples.npz exists.",
    )
    parser.add_argument(
        "--zoom-panels",
        type=int,
        default=None,
        help="Series count for last-step zoom figure (default: same as --max-panels).",
    )
    parser.add_argument(
        "--zoom-margin-frac",
        type=float,
        default=0.45,
        help="Padding around min bbox of {true, mean, samples, CV, context tail} for zoom plots.",
    )
    parser.add_argument(
        "--context-tail-in-zoom",
        type=int,
        default=6,
        help="How many last context points to draw inside the zoom (orientation).",
    )
    parser.add_argument(
        "--max-pair-panels",
        type=int,
        default=8,
        help="Max rows in viz_full_and_zoom.png (each row is tall); avoids huge PNGs.",
    )
    args = parser.parse_args()

    zoom_panels = args.zoom_panels if args.zoom_panels is not None else args.max_panels

    run_dir = Path(args.run_dir).resolve()
    meta_path = run_dir / "preprocess_metadata.json"
    npz_path = run_dir / "train_test_trajectories.npz"
    fc_path = run_dir / "forecast_samples.npz"
    metrics_path = run_dir / "metrics.json"

    if not meta_path.is_file() or not npz_path.is_file():
        raise SystemExit(f"Need {meta_path} and {npz_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prediction_length = int(meta["prediction_length"])
    context_length = int(meta["context_length"])
    state_repr = meta.get("state_repr", "absolute")
    denorm = bool(meta.get("denorm_metrics", False))

    raw = np.load(npz_path, allow_pickle=True)
    base_test = list(raw["base_test_trajectories"])
    mean = np.asarray(raw["mean"], dtype=np.float32)
    std = np.asarray(raw["std"], dtype=np.float32)
    test_z = list(raw["test_trajectories"])

    metrics = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CV errors in physical space (always available) ---
    cv_dx, cv_dy = [], []
    true_x, true_y = [], []
    for traj in base_test:
        t = np.asarray(traj, dtype=np.float32)
        prev = t[-prediction_length - 1].copy()
        pred_cv = cv_predict_one_step(prev)
        true = t[-prediction_length:]
        for k in range(prediction_length):
            err = pred_cv[:2] - true[k, :2]
            cv_dx.append(float(err[0]))
            cv_dy.append(float(err[1]))
            true_x.append(float(true[k, 0]))
            true_y.append(float(true[k, 1]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax0, ax1, ax2 = axes

    n_sp = min(args.max_spaghetti, len(base_test))
    for i in range(n_sp):
        t = np.asarray(base_test[i], dtype=np.float32)
        ax0.plot(t[:, 0], t[:, 1], alpha=0.25, linewidth=0.8, color="C0")
    ax0.set_title(f"GT test paths (n={n_sp})")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.grid(True, alpha=0.3)

    ax1.scatter(cv_dx, cv_dy, s=4, alpha=0.35, c="green")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(0, color="k", linewidth=0.5)
    ax1.set_title("CV baseline position error (pred−true), m")
    ax1.set_xlabel("Δx")
    ax1.set_ylabel("Δy")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(true_x, true_y, s=4, alpha=0.25, c="gray")
    ax2.set_title("Test-set true positions at forecast step(s)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(run_dir.name, fontsize=10)
    fig.tight_layout()
    p0 = out_dir / "viz_gt_cv_coverage.png"
    fig.savefig(p0, dpi=150)
    plt.close(fig)

    # --- Model sample clouds (optional) ---
    p1 = out_dir / "viz_model_vs_gt.png"
    p_zoom = out_dir / "viz_last_step_zoom.png"
    p_pair = out_dir / "viz_full_and_zoom.png"
    if fc_path.is_file():
        fc = np.load(fc_path, allow_pickle=True)
        samples_obj = fc["samples"]
        n_series = min(len(samples_obj), len(base_test), args.max_panels)
        ncol = 4
        nrow = int(np.ceil(n_series / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
        axes_flat = np.atleast_1d(axes).ravel()

        for idx in range(n_series):
            ax = axes_flat[idx]
            samples = np.asarray(samples_obj[idx], dtype=np.float32)
            base = np.asarray(base_test[idx], dtype=np.float32)
            tz = np.asarray(test_z[idx], dtype=np.float32)
            bundle = series_forecast_bundle(
                base,
                tz,
                samples,
                mean=mean,
                std=std,
                context_length=context_length,
                prediction_length=prediction_length,
                state_repr=state_repr,
                denorm=denorm,
            )
            ctx_p = bundle["ctx_p"]
            true_p = bundle["true_p"]
            mean_abs = bundle["mean_abs"]
            samp_abs = bundle["samp_abs"]

            ax.plot(ctx_p[:, 0], ctx_p[:, 1], "b-", linewidth=1.0, label="context")
            ax.scatter(true_p[:, 0], true_p[:, 1], c="k", s=20, marker="x", label="true", zorder=5)
            ax.scatter(mean_abs[:, 0], mean_abs[:, 1], c="red", s=12, label="mean", zorder=4)
            ax.scatter(
                samp_abs[:, :, 0].ravel(),
                samp_abs[:, :, 1].ravel(),
                c="orange",
                s=2,
                alpha=0.25,
                label="samples",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"series {idx}")
            if idx == 0:
                ax.legend(fontsize=6, loc="upper right")

        for j in range(n_series, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"{run_dir.name} — model samples (denorm={denorm})", fontsize=10)
        fig.tight_layout()
        fig.savefig(p1, dpi=150)
        plt.close(fig)

        # --- Zoom: last forecast step only (per-series) ---
        n_zoom = min(len(samples_obj), len(base_test), zoom_panels)
        ncol_z = 4
        nrow_z = int(np.ceil(n_zoom / ncol_z))
        fig_z, axes_z = plt.subplots(nrow_z, ncol_z, figsize=(3.6 * ncol_z, 3.6 * nrow_z))
        axes_z_flat = np.atleast_1d(axes_z).ravel()

        last_step_label = f"step {prediction_length - 1}" if prediction_length > 1 else "single step"

        for idx in range(n_zoom):
            ax = axes_z_flat[idx]
            samples = np.asarray(samples_obj[idx], dtype=np.float32)
            base = np.asarray(base_test[idx], dtype=np.float32)
            tz = np.asarray(test_z[idx], dtype=np.float32)
            bundle = series_forecast_bundle(
                base,
                tz,
                samples,
                mean=mean,
                std=std,
                context_length=context_length,
                prediction_length=prediction_length,
                state_repr=state_repr,
                denorm=denorm,
            )
            ctx_p = bundle["ctx_p"]
            true_p = bundle["true_p"]
            mean_abs = bundle["mean_abs"]
            samp_abs = bundle["samp_abs"]
            cv_pred = bundle["cv_pred"]

            tail = max(1, min(args.context_tail_in_zoom, len(ctx_p)))
            ax.plot(ctx_p[-tail:, 0], ctx_p[-tail:, 1], "b-", linewidth=1.2, label="context tail")
            ax.scatter(ctx_p[-1, 0], ctx_p[-1, 1], c="blue", s=35, marker="o", zorder=4, label="ctx end")

            ax.scatter(
                samp_abs[:, -1, 0],
                samp_abs[:, -1, 1],
                c="orange",
                s=8,
                alpha=0.35,
                label="samples @ last",
            )
            ax.scatter(mean_abs[-1, 0], mean_abs[-1, 1], c="red", s=70, marker="*", zorder=6, label="mean")
            ax.scatter(true_p[-1, 0], true_p[-1, 1], c="k", s=60, marker="x", linewidths=1.5, zorder=7, label="true")
            ax.scatter(cv_pred[0], cv_pred[1], c="green", s=45, marker="D", zorder=5, label="CV 1-step")

            ax.plot(
                [mean_abs[-1, 0], true_p[-1, 0]],
                [mean_abs[-1, 1], true_p[-1, 1]],
                "r--",
                linewidth=1.0,
                alpha=0.8,
            )

            tpos = true_p[-1, :2]
            mpos = mean_abs[-1, :2]
            cpos = cv_pred[:2]
            err_m = float(np.linalg.norm(mpos - tpos))
            err_cv = float(np.linalg.norm(cpos - tpos))
            spread = float(np.std(np.linalg.norm(samp_abs[:, -1, :2] - tpos.reshape(1, 2), axis=1)))

            x0, x1, y0, y1 = zoom_limits_around_forecast(
                bundle,
                context_tail=args.context_tail_in_zoom,
                margin_frac=args.zoom_margin_frac,
            )
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.35)
            ax.set_title(f"series {idx} — zoom {last_step_label}", fontsize=9)
            txt = f"|mean−true|={err_m:.3g} m\n|CV−true|={err_cv:.3g} m\nstd|sample−true|={spread:.3g}"
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
            )
            if idx == 0:
                ax.legend(fontsize=5, loc="lower right")

        for j in range(n_zoom, len(axes_z_flat)):
            axes_z_flat[j].set_visible(False)

        fig_z.suptitle(
            f"{run_dir.name} — last forecast step (physical m), margin_frac={args.zoom_margin_frac}",
            fontsize=10,
        )
        fig_z.tight_layout()
        fig_z.savefig(p_zoom, dpi=180)
        plt.close(fig_z)

        # --- Side-by-side: full trajectory context vs same zoom ---
        ncol_p = 2
        n_pair = min(n_zoom, args.max_pair_panels)
        nrow_p = n_pair
        fig_p, axes_p = plt.subplots(nrow_p, ncol_p, figsize=(7.0, 2.8 * nrow_p))
        if nrow_p == 1:
            axes_p = axes_p.reshape(1, -1)

        for row, idx in enumerate(range(n_pair)):
            samples = np.asarray(samples_obj[idx], dtype=np.float32)
            base = np.asarray(base_test[idx], dtype=np.float32)
            tz = np.asarray(test_z[idx], dtype=np.float32)
            bundle = series_forecast_bundle(
                base,
                tz,
                samples,
                mean=mean,
                std=std,
                context_length=context_length,
                prediction_length=prediction_length,
                state_repr=state_repr,
                denorm=denorm,
            )
            ctx_p = bundle["ctx_p"]
            true_p = bundle["true_p"]
            mean_abs = bundle["mean_abs"]
            samp_abs = bundle["samp_abs"]
            cv_pred = bundle["cv_pred"]

            ax_full = axes_p[row, 0]
            ax_full.plot(ctx_p[:, 0], ctx_p[:, 1], "b-", linewidth=1.0)
            ax_full.scatter(true_p[:, 0], true_p[:, 1], c="k", s=18, marker="x", zorder=5)
            ax_full.scatter(mean_abs[:, 0], mean_abs[:, 1], c="red", s=10, zorder=4)
            ax_full.scatter(samp_abs[:, :, 0].ravel(), samp_abs[:, :, 1].ravel(), c="orange", s=1, alpha=0.2)
            ax_full.set_title(f"series {idx} full context")
            ax_full.set_aspect("equal", adjustable="box")
            ax_full.grid(True, alpha=0.3)

            ax_z = axes_p[row, 1]
            tail = max(1, min(args.context_tail_in_zoom, len(ctx_p)))
            ax_z.plot(ctx_p[-tail:, 0], ctx_p[-tail:, 1], "b-", linewidth=1.2)
            ax_z.scatter(ctx_p[-1, 0], ctx_p[-1, 1], c="blue", s=30, marker="o", zorder=4)
            ax_z.scatter(samp_abs[:, -1, 0], samp_abs[:, -1, 1], c="orange", s=8, alpha=0.35)
            ax_z.scatter(mean_abs[-1, 0], mean_abs[-1, 1], c="red", s=60, marker="*", zorder=6)
            ax_z.scatter(true_p[-1, 0], true_p[-1, 1], c="k", s=50, marker="x", linewidths=1.5, zorder=7)
            ax_z.scatter(cv_pred[0], cv_pred[1], c="green", s=40, marker="D", zorder=5)
            ax_z.plot(
                [mean_abs[-1, 0], true_p[-1, 0]],
                [mean_abs[-1, 1], true_p[-1, 1]],
                "r--",
                linewidth=1.0,
                alpha=0.8,
            )
            x0, x1, y0, y1 = zoom_limits_around_forecast(
                bundle,
                context_tail=args.context_tail_in_zoom,
                margin_frac=args.zoom_margin_frac,
            )
            ax_z.set_xlim(x0, x1)
            ax_z.set_ylim(y0, y1)
            ax_z.set_title(f"series {idx} zoom ({last_step_label})")
            ax_z.set_aspect("equal", adjustable="box")
            ax_z.grid(True, alpha=0.35)

        pair_note = f" (first {n_pair} of {n_zoom} zoom series)" if n_pair < n_zoom else ""
        fig_p.suptitle(f"{run_dir.name} — full context vs last-step zoom{pair_note}", fontsize=10)
        fig_p.tight_layout()
        fig_p.savefig(p_pair, dpi=160)
        plt.close(fig_p)

    else:
        p1 = None
        p_zoom = None
        p_pair = None

    # --- Error distribution comparison text ---
    lines = [
        f"Run: {run_dir}",
        f"prediction_length={prediction_length}, context_length={context_length}, denorm_metrics={denorm}, state_repr={state_repr}",
        "",
        "CV baseline position error stats (m):",
        f"  mean |err| = {np.mean(np.hypot(cv_dx, cv_dy)):.4f}",
        f"  std  |err| = {np.std(np.hypot(cv_dx, cv_dy)):.4f}",
    ]
    if metrics:
        lines += [
            "",
            "From metrics.json (if present):",
            f"  ADE={metrics.get('ADE')}, CV_ADE={metrics.get('CV_ADE')}, ratio={metrics.get('Model_vs_CV_ADE_Ratio')}",
        ]
    lines += [
        "",
        f"Saved: {p0}",
    ]
    if p1:
        lines.append(f"Saved: {p1}")
        lines.append(f"Saved: {p_zoom}")
        lines.append(f"Saved: {p_pair}")
    else:
        lines += [
            "Model sample figure skipped (no forecast_samples.npz).",
            "Re-run training once with:  --save-forecast-samples",
        ]

    report = out_dir / "viz_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
