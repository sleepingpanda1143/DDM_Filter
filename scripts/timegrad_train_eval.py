#!/usr/bin/env python3
"""
Train and evaluate GluonTS-style TimeGrad (diffusers + pts TimeGradEstimator) on
pre-built dataset folders that contain ``preprocess_metadata.json`` (radar / motion runs).

Example:
  ./scripts/run_timegrad.sh smoke
  ./scripts/run_timegrad.sh train --preset radar_viz_debug --max-epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def _denorm_rmse(
    forecasts: List[Any],
    tss: List[pd.DataFrame],
    mean: np.ndarray,
    std: np.ndarray,
    prediction_length: int,
) -> Optional[float]:
    """Mean RMSE over series after denormalizing with dataset mean/std (median forecast)."""
    if mean is None or std is None:
        return None
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    errs = []
    for fc, ts in zip(forecasts, tss):
        pred = fc.quantile(0.5)
        pred = np.asarray(pred, dtype=np.float64)
        obs = np.asarray(ts.to_numpy(copy=False), dtype=np.float64)
        obs = obs[-prediction_length:]
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        p = pred * std + mean
        o = obs * std + mean
        errs.append(np.sqrt(np.mean((p - o) ** 2)))
    return float(np.mean(errs)) if errs else None


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
            shuffle_buffer_length=max(100, args.batch_size * 4),
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
        denorm = None
        if meta.get("denorm_metrics") and mean is not None and std is not None:
            denorm = _denorm_rmse(
                forecasts, tss, mean, std, prediction_length=prediction_length
            )

        out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "timegrad_autoruns"
        out_dir = out_dir / f"{int(time.time())}_{args.encoder_type}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "dataset_dir": str(dataset_dir),
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-batches-per-epoch", type=int, default=None)
    parser.add_argument("--num-parallel-samples", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--eval-num-samples", type=int, default=50)

    args = parser.parse_args()

    if args.accelerator == "auto":
        args.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    if args.mode == "smoke":
        args.max_epochs = 2 if args.max_epochs is None else args.max_epochs
        args.num_batches_per_epoch = 25 if args.num_batches_per_epoch is None else args.num_batches_per_epoch
        args.batch_size = min(int(args.batch_size), 16)
        args.num_inference_steps = min(int(args.num_inference_steps), 25)
        args.eval_num_samples = min(int(args.eval_num_samples), 20)
    else:
        args.max_epochs = 15 if args.max_epochs is None else args.max_epochs
        args.num_batches_per_epoch = 80 if args.num_batches_per_epoch is None else args.num_batches_per_epoch

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
                line += f"  denorm_RMSE(median_fc)={summary['denorm_median_rmse']:.6g}"
            print(line, flush=True)
        except Exception as e:
            print(f"FAILED {name}: {e}", flush=True)
            rows.append({"dataset_dir": str(dataset_dir), "error": str(e)})

    out = REPO_ROOT / "dataset" / "_timegrad_last_run_summary.json"
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote combined summary to {out}", flush=True)


if __name__ == "__main__":
    main()
