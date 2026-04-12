import argparse
import json
import time
import sys
import warnings
from pathlib import Path

if "--suppress-warnings" in sys.argv:
    warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
from pts.dataset.loader import TransformedIterableDataset
from pts import Trainer
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)
# PyTorchEstimator + Trainer path (not Lightning `estimator.py`, which requires `scheduler`, etc.)
from pts.model.time_grad.time_grad_estimator import TimeGradEstimator


class RecordingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_losses = []
        self.epoch_times_sec = []

    def __call__(self, net: nn.Module, train_iter, validation_iter=None) -> None:
        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        self.epoch_losses = []
        self.epoch_times_sec = []

        for epoch_no in range(self.epochs):
            tic = time.time()
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = [v.to(self.device) for v in data_entry.values()]
                    output = net(*inputs)
                    loss = output[0] if isinstance(output, (list, tuple)) else output
                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )

                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break

            toc = time.time()
            self.epoch_losses.append(float(avg_epoch_loss))
            self.epoch_times_sec.append(float(toc - tic))


def load_clean_trajectories(npz_path: Path) -> list:
    data = np.load(npz_path, allow_pickle=True)
    trajectories = data["trajectories"]
    cleaned = []
    for trajectory in trajectories:
        array = np.asarray(trajectory, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 4:
            continue
        if not np.isfinite(array).all():
            continue
        cleaned.append(array)
    return cleaned


def split_trajectories(trajectories: list, train_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(trajectories))
    rng.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [trajectories[i] for i in train_idx]
    test = [trajectories[i] for i in test_idx]
    return train, test, train_idx.tolist(), test_idx.tolist()


def filter_short(trajectories: list, min_length: int) -> list:
    return [trajectory for trajectory in trajectories if len(trajectory) >= min_length]


def compute_zscore_stats(train_trajectories: list):
    concat = np.concatenate(train_trajectories, axis=0)
    mean = concat.mean(axis=0)
    std = concat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(trajectories: list, mean: np.ndarray, std: np.ndarray) -> list:
    normalized = []
    for trajectory in trajectories:
        normalized.append(((trajectory - mean) / std).astype(np.float32))
    return normalized


def apply_position_origin(trajectories: list, mode: str) -> list:
    if mode == "none":
        return [trajectory.astype(np.float32) for trajectory in trajectories]

    adjusted = []
    for trajectory in trajectories:
        values = trajectory.astype(np.float32).copy()
        if mode == "first":
            origin = values[0, :2].copy()
            values[:, :2] = values[:, :2] - origin.reshape(1, 2)
        else:
            raise ValueError(f"Unknown position origin mode: {mode}")
        adjusted.append(values)
    return adjusted


def apply_state_representation(trajectories: list, mode: str) -> list:
    if mode == "absolute":
        return [trajectory.astype(np.float32) for trajectory in trajectories]

    represented = []
    for trajectory in trajectories:
        values = trajectory.astype(np.float32)
        if mode == "delta":
            delta = np.zeros_like(values, dtype=np.float32)
            delta[1:] = values[1:] - values[:-1]
            represented.append(delta)
        else:
            raise ValueError(f"Unknown state representation: {mode}")
    return represented


def build_list_dataset(trajectories: list, freq: str):
    entries = []
    start = pd.Timestamp("2000-01-01 00:00:00")
    for trajectory in trajectories:
        entries.append({"start": start, "target": trajectory.T})
    return ListDataset(entries, freq=freq, one_dim_target=False)


def calc_input_size(target_dim: int, freq: str) -> int:
    lags = lags_for_fourier_time_features_from_frequency(freq)
    time_features = fourier_time_features_from_frequency(freq)
    return target_dim * len(lags) + target_dim + len(time_features)


def infer_input_size_from_dataset(
    train_ds,
    freq: str,
    prediction_length: int,
    context_length: int,
    target_dim: int,
):
    probe_estimator = TimeGradEstimator(
        target_dim=target_dim,
        prediction_length=prediction_length,
        context_length=context_length,
        cell_type="GRU",
        input_size=1,
        freq=freq,
        loss_type="l2",
        scaling=True,
        diff_steps=10,
        beta_end=0.1,
        beta_schedule="linear",
        trainer=Trainer(
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=1e-3,
            num_batches_per_epoch=1,
            batch_size=1,
        ),
    )

    transform = probe_estimator.create_transformation() + probe_estimator.create_instance_splitter("training")
    transformed = TransformedIterableDataset(
        dataset=train_ds,
        transform=transform,
        is_train=True,
        cache_data=False,
    )
    sample = next(iter(transformed))
    time_feat_key = "past_time_feat" if "past_time_feat" in sample else "past_feat_time"
    time_feat_size = int(sample[time_feat_key].shape[-1])
    input_size = target_dim * len(probe_estimator.lags_seq) + target_dim + time_feat_size
    return input_size


def denormalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return array * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def extract_target_last_horizon(target_obj, prediction_length: int, target_dim: int) -> np.ndarray:
    values = np.asarray(target_obj.values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values = values[-prediction_length:, :target_dim]
    return values


def extract_point_forecast(forecast_obj) -> np.ndarray:
    samples = np.asarray(forecast_obj.samples, dtype=np.float32)
    if samples.ndim != 3:
        raise RuntimeError(f"Unexpected forecast samples shape: {samples.shape}")
    return samples.mean(axis=0)


def compute_tracking_metrics(
    forecasts,
    transformed_trajectories: list,
    base_trajectories: list,
    prediction_length: int,
    mean: np.ndarray,
    std: np.ndarray,
    denorm: bool,
    state_repr: str,
):
    pos_distances = []
    all_pos_pred = []
    all_pos_true = []
    all_vel_pred = []
    all_vel_true = []

    for forecast_obj, transformed_values, base_values in zip(forecasts, transformed_trajectories, base_trajectories):
        pred = extract_point_forecast(forecast_obj)
        transformed_values = transformed_values.astype(np.float32)
        base_values = base_values.astype(np.float32)

        true = transformed_values[-prediction_length:, : pred.shape[-1]]

        pred = pred.reshape(prediction_length, -1)
        true = true.reshape(prediction_length, -1)

        if denorm:
            pred = denormalize(pred.reshape(1, prediction_length, -1), mean, std).reshape(prediction_length, -1)
            true = denormalize(true.reshape(1, prediction_length, -1), mean, std).reshape(prediction_length, -1)

        if state_repr == "delta":
            anchor = base_values[-prediction_length - 1].copy()
            pred_abs = []
            running = anchor.copy()
            for step in range(prediction_length):
                running = running + pred[step]
                pred_abs.append(running.copy())
            pred = np.asarray(pred_abs, dtype=np.float32)
            true = base_values[-prediction_length:, : pred.shape[-1]]

        pos_pred = pred[:, :2]
        pos_true = true[:, :2]
        vel_pred = pred[:, 2:4]
        vel_true = true[:, 2:4]

        dist = np.linalg.norm(pos_pred - pos_true, axis=1)
        pos_distances.append(dist)

        all_pos_pred.append(pos_pred)
        all_pos_true.append(pos_true)
        all_vel_pred.append(vel_pred)
        all_vel_true.append(vel_true)

    pos_distances = np.concatenate(pos_distances, axis=0)
    all_pos_pred = np.concatenate(all_pos_pred, axis=0)
    all_pos_true = np.concatenate(all_pos_true, axis=0)
    all_vel_pred = np.concatenate(all_vel_pred, axis=0)
    all_vel_true = np.concatenate(all_vel_true, axis=0)

    rmse_pos = float(np.sqrt(np.mean((all_pos_pred - all_pos_true) ** 2)))
    rmse_vel = float(np.sqrt(np.mean((all_vel_pred - all_vel_true) ** 2)))
    ade = float(np.mean(pos_distances))

    if prediction_length > 1:
        fde_list = []
        for forecast_obj, transformed_values, base_values in zip(forecasts, transformed_trajectories, base_trajectories):
            pred = extract_point_forecast(forecast_obj).reshape(prediction_length, -1)
            transformed_values = transformed_values.astype(np.float32)
            base_values = base_values.astype(np.float32)
            true = transformed_values[-prediction_length:, : pred.shape[-1]].reshape(prediction_length, -1)
            if denorm:
                pred = denormalize(pred.reshape(1, prediction_length, -1), mean, std).reshape(prediction_length, -1)
                true = denormalize(true.reshape(1, prediction_length, -1), mean, std).reshape(prediction_length, -1)

            if state_repr == "delta":
                anchor = base_values[-prediction_length - 1].copy()
                pred_abs = []
                running = anchor.copy()
                for step in range(prediction_length):
                    running = running + pred[step]
                    pred_abs.append(running.copy())
                pred = np.asarray(pred_abs, dtype=np.float32)
                true = base_values[-prediction_length:, : pred.shape[-1]]

            fde_list.append(np.linalg.norm(pred[-1, :2] - true[-1, :2]))
        fde = float(np.mean(fde_list))
    else:
        fde = ade

    return {
        "ADE": ade,
        "FDE": fde,
        "RMSE_pos": rmse_pos,
        "RMSE_vel": rmse_vel,
    }


def compute_naive_cv_metrics(
    trajectories: list,
    prediction_length: int,
):
    pos_distances = []
    all_pos_pred = []
    all_pos_true = []
    all_vel_pred = []
    all_vel_true = []

    for trajectory in trajectories:
        values = trajectory.astype(np.float32)

        prev = values[-prediction_length - 1].copy()
        pred_steps = []
        for _ in range(prediction_length):
            next_state = prev.copy()
            next_state[0] = prev[0] + prev[2]
            next_state[1] = prev[1] + prev[3]
            next_state[2] = prev[2]
            next_state[3] = prev[3]
            pred_steps.append(next_state)
            prev = next_state

        pred = np.asarray(pred_steps, dtype=np.float32)
        true = values[-prediction_length:]

        pos_pred = pred[:, :2]
        pos_true = true[:, :2]
        vel_pred = pred[:, 2:4]
        vel_true = true[:, 2:4]

        dist = np.linalg.norm(pos_pred - pos_true, axis=1)
        pos_distances.append(dist)
        all_pos_pred.append(pos_pred)
        all_pos_true.append(pos_true)
        all_vel_pred.append(vel_pred)
        all_vel_true.append(vel_true)

    pos_distances = np.concatenate(pos_distances, axis=0)
    all_pos_pred = np.concatenate(all_pos_pred, axis=0)
    all_pos_true = np.concatenate(all_pos_true, axis=0)
    all_vel_pred = np.concatenate(all_vel_pred, axis=0)
    all_vel_true = np.concatenate(all_vel_true, axis=0)

    rmse_pos = float(np.sqrt(np.mean((all_pos_pred - all_pos_true) ** 2)))
    rmse_vel = float(np.sqrt(np.mean((all_vel_pred - all_vel_true) ** 2)))
    ade = float(np.mean(pos_distances))

    if prediction_length > 1:
        fde_list = []
        for trajectory in trajectories:
            values = trajectory.astype(np.float32)
            prev = values[-prediction_length - 1].copy()
            pred_steps = []
            for _ in range(prediction_length):
                next_state = prev.copy()
                next_state[0] = prev[0] + prev[2]
                next_state[1] = prev[1] + prev[3]
                next_state[2] = prev[2]
                next_state[3] = prev[3]
                pred_steps.append(next_state)
                prev = next_state
            pred = np.asarray(pred_steps, dtype=np.float32)
            true = values[-prediction_length:]
            fde_list.append(np.linalg.norm(pred[-1, :2] - true[-1, :2]))
        fde = float(np.mean(fde_list))
    else:
        fde = ade

    return {
        "CV_ADE": ade,
        "CV_FDE": fde,
        "CV_RMSE_pos": rmse_pos,
        "CV_RMSE_vel": rmse_vel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-npz",
        type=str,
        default="/home/xiongmaoren/kalman_net/Radar/data/trajectory_data.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/xiongmaoren/kalman_net/DDM_Timeseries_Forecast/dataset/radar_tracking_clean",
    )
    parser.add_argument("--freq", type=str, default="1min")
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-batches-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for training transforms (0=main thread; try 4–8 to reduce GPU idle on fast GPUs).",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--max-train-trajectories",
        type=int,
        default=300,
        help="Cap training trajectories; use 0 for no cap (use full train split).",
    )
    parser.add_argument("--max-test-trajectories", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--denorm-metrics", action="store_true", help="Evaluate ADE/FDE/RMSE in original physical units.")
    parser.add_argument("--suppress-warnings", action="store_true", help="Suppress warnings to reduce log noise.")
    parser.add_argument("--save-train-loss", action="store_true", help="Save epoch-level training loss and epoch time to JSON.")
    parser.add_argument("--state-repr", type=str, default="absolute", choices=["absolute", "delta"], help="Target representation for modeling.")
    parser.add_argument("--position-origin", type=str, default="none", choices=["none", "first"], help="Optional per-trajectory position origin adjustment.")
    parser.add_argument(
        "--no-timegrad-scaling",
        action="store_true",
        help="Disable TimeGrad MeanScaler (data are already z-scored; avoids double scaling).",
    )
    parser.add_argument("--num-layers", type=int, default=2, help="RNN depth in TimeGrad.")
    parser.add_argument("--num-cells", type=int, default=40, help="RNN hidden size in TimeGrad.")
    parser.add_argument("--diff-steps", type=int, default=100, help="Diffusion steps in TimeGrad training/sampling.")
    parser.add_argument(
        "--save-forecast-samples",
        action="store_true",
        help="Save per-series diffusion sample tensors to forecast_samples.npz for offline visualization.",
    )
    args = parser.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_clean_trajectories(Path(args.input_npz))
    min_length = args.context_length + args.prediction_length + 1
    trajectories = filter_short(trajectories, min_length=min_length)

    train_traj, test_traj, train_idx, test_idx = split_trajectories(
        trajectories, train_ratio=args.train_ratio, seed=args.seed
    )

    if args.max_train_trajectories and args.max_train_trajectories > 0:
        train_traj = train_traj[: args.max_train_trajectories]
    if args.max_test_trajectories and args.max_test_trajectories > 0:
        test_traj = test_traj[: args.max_test_trajectories]

    if len(train_traj) == 0 or len(test_traj) == 0:
        raise RuntimeError("Train/Test trajectories are empty after filtering and slicing.")

    base_train_traj = apply_position_origin(train_traj, mode=args.position_origin)
    base_test_traj = apply_position_origin(test_traj, mode=args.position_origin)

    repr_train_traj = apply_state_representation(base_train_traj, mode=args.state_repr)
    repr_test_traj = apply_state_representation(base_test_traj, mode=args.state_repr)

    mean, std = compute_zscore_stats(repr_train_traj)
    train_traj = apply_zscore(repr_train_traj, mean, std)
    test_traj = apply_zscore(repr_test_traj, mean, std)

    train_ds = build_list_dataset(train_traj, freq=args.freq)
    test_ds = build_list_dataset(test_traj, freq=args.freq)

    target_dim = 4
    input_size_formula = calc_input_size(target_dim=target_dim, freq=args.freq)
    input_size = infer_input_size_from_dataset(
        train_ds=train_ds,
        freq=args.freq,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        target_dim=target_dim,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    device = torch.device(args.device)

    print(
        f"Runtime: device={device}, cuda_available={torch.cuda.is_available()}, "
        f"torch_threads={torch.get_num_threads()}"
    )
    total_steps = args.epochs * args.num_batches_per_epoch
    print(f"Planned optimization steps: {total_steps}")

    trainer = RecordingTrainer(
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_batches_per_epoch=args.num_batches_per_epoch,
        batch_size=args.batch_size,
    )

    estimator = TimeGradEstimator(
        target_dim=target_dim,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        cell_type="GRU",
        num_layers=args.num_layers,
        num_cells=args.num_cells,
        input_size=input_size,
        freq=args.freq,
        loss_type="l2",
        scaling=not args.no_timegrad_scaling,
        diff_steps=args.diff_steps,
        beta_end=0.1,
        beta_schedule="linear",
        trainer=trainer,
    )

    print(f"Prepared trajectories: total={len(trajectories)}, train={len(train_traj)}, test={len(test_traj)}")
    print(
        f"Using freq={args.freq}, target_dim={target_dim}, "
        f"input_size(inferred)={input_size}, input_size(formula)={input_size_formula}"
    )

    predictor = estimator.train(train_ds, num_workers=args.num_workers)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=args.num_samples,
    )

    forecasts = list(forecast_it)
    targets = list(ts_it)

    if args.save_forecast_samples:
        samples_list = [np.asarray(f.samples, dtype=np.float32) for f in forecasts]
        np.savez_compressed(
            output_dir / "forecast_samples.npz",
            samples=np.array(samples_list, dtype=object),
        )
        print(f"Saved forecast sample arrays: {output_dir / 'forecast_samples.npz'}")

    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20) / 20.0)[1:],
        target_agg_funcs={"sum": np.sum},
    )
    agg_metrics, _ = evaluator(targets, forecasts, num_series=len(test_traj))

    metrics = {
        "CRPS": float(agg_metrics["mean_wQuantileLoss"]),
        "ND": float(agg_metrics["ND"]),
        "NRMSE": float(agg_metrics["NRMSE"]),
        "CRPS_Sum": float(agg_metrics["m_sum_mean_wQuantileLoss"]),
        "ND_Sum": float(agg_metrics["m_sum_ND"]),
        "NRMSE_Sum": float(agg_metrics["m_sum_NRMSE"]),
    }

    tracking_metrics = compute_tracking_metrics(
        forecasts=forecasts,
        transformed_trajectories=test_traj,
        base_trajectories=base_test_traj,
        prediction_length=args.prediction_length,
        mean=mean,
        std=std,
        denorm=args.denorm_metrics,
        state_repr=args.state_repr,
    )
    metrics.update(tracking_metrics)

    naive_cv_metrics = compute_naive_cv_metrics(
        trajectories=base_test_traj,
        prediction_length=args.prediction_length,
    )
    metrics.update(naive_cv_metrics)

    metrics["Model_vs_CV_ADE_Ratio"] = float(metrics["ADE"] / max(metrics["CV_ADE"], 1e-8))
    metrics["Model_vs_CV_RMSE_pos_Ratio"] = float(metrics["RMSE_pos"] / max(metrics["CV_RMSE_pos"], 1e-8))

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    metadata = {
        "input_npz": str(Path(args.input_npz).resolve()),
        "num_total_trajectories": len(trajectories),
        "num_train": len(train_traj),
        "num_test": len(test_traj),
        "train_indices": train_idx,
        "test_indices": test_idx,
        "freq": args.freq,
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "target_dim": target_dim,
        "input_size": input_size,
        "input_size_formula": input_size_formula,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "seed": args.seed,
        "denorm_metrics": args.denorm_metrics,
        "suppress_warnings": args.suppress_warnings,
        "save_train_loss": args.save_train_loss,
        "state_repr": args.state_repr,
        "position_origin": args.position_origin,
        "timegrad_scaling": not args.no_timegrad_scaling,
        "num_layers": args.num_layers,
        "num_cells": args.num_cells,
        "diff_steps": args.diff_steps,
        "save_forecast_samples": args.save_forecast_samples,
        "num_workers": args.num_workers,
    }

    with open(output_dir / "preprocess_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    if args.save_train_loss:
        train_loss_payload = {
            "epoch_losses": trainer.epoch_losses,
            "epoch_times_sec": trainer.epoch_times_sec,
        }
        with open(output_dir / "train_loss.json", "w", encoding="utf-8") as file:
            json.dump(train_loss_payload, file, indent=2, ensure_ascii=False)

    np.savez_compressed(
        output_dir / "train_test_trajectories.npz",
        train_trajectories=np.array(train_traj, dtype=object),
        test_trajectories=np.array(test_traj, dtype=object),
        base_train_trajectories=np.array(base_train_traj, dtype=object),
        base_test_trajectories=np.array(base_test_traj, dtype=object),
        mean=mean,
        std=std,
    )

    model_dir = output_dir / "predictor"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        predictor.serialize(model_dir)
    except RuntimeError as exc:
        print(
            "Warning: predictor.serialize() failed (often GluonTS JSON serde + torch.device). "
            f"Metrics and npz are still saved. Error: {exc}"
        )

    print(f"Saved metadata: {output_dir / 'preprocess_metadata.json'}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    if args.save_train_loss:
        print(f"Saved train loss: {output_dir / 'train_loss.json'}")
    print(f"Saved processed trajectories: {output_dir / 'train_test_trajectories.npz'}")
    print(f"Saved predictor: {model_dir}")


if __name__ == "__main__":
    main()
