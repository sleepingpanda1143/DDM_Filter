#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def apply_zscore(trajectories: List[np.ndarray], mean: np.ndarray, std: np.ndarray) -> List[np.ndarray]:
    out = []
    for tr in trajectories:
        out.append(((tr - mean) / std).astype(np.float32))
    return out


def load_paired_measurements(npz_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    traj_raw = data["trajectories"]
    has_meas = "measurements_pos" in data
    out_t: List[np.ndarray] = []
    out_m: List[np.ndarray] = []
    for idx in range(len(traj_raw)):
        traj = np.asarray(traj_raw[idx], dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] not in (4, 6) or not np.isfinite(traj).all():
            continue
        if has_meas:
            meas = np.asarray(data["measurements_pos"][idx], dtype=np.float32)
        else:
            k = 3 if traj.shape[1] >= 6 else 2
            meas = traj[:, :k].astype(np.float32, copy=False)
        if meas.ndim != 2 or meas.shape[0] != traj.shape[0] or meas.shape[1] not in (2, 3):
            continue
        if not np.isfinite(meas).all():
            continue
        out_t.append(traj)
        out_m.append(meas.astype(np.float32, copy=False))
    return out_t, out_m


def subtract_first_position_from_measurements(
    measurements: List[np.ndarray],
    raw_trajectories: List[np.ndarray],
    position_origin_mode: str,
) -> List[np.ndarray]:
    if position_origin_mode == "none":
        return [m.copy() for m in measurements]
    if position_origin_mode != "first":
        raise ValueError(position_origin_mode)
    adjusted = []
    for meas, traj in zip(measurements, raw_trajectories):
        k = int(meas.shape[1])
        origin = traj[0, :k].astype(np.float32, copy=False)
        adjusted.append((meas - origin.reshape(1, -1)).astype(np.float32))
    return adjusted


def _denorm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std.reshape(1, -1) + mean.reshape(1, -1)


def _read_sigma(npz_path: Path) -> Optional[float]:
    sidecar = npz_path.with_suffix('.json')
    if not sidecar.is_file():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
    except Exception:
        return None
    v = meta.get('measurement_sigma_m')
    return None if v is None else float(v)


def _cv_diff_predict(z_tm2: np.ndarray, z_tm1: np.ndarray) -> np.ndarray:
    # one-step extrapolation to current t using two previous measurements
    return z_tm1 + (z_tm1 - z_tm2)


def _kf_cv_positions(meas: np.ndarray, q_acc: float, r_var: float) -> np.ndarray:
    T = len(meas)
    out = np.full((T, 3), np.nan, dtype=np.float64)
    if T == 0:
        return out
    dt = 1.0
    F = np.block([[np.eye(3), dt*np.eye(3)], [np.zeros((3,3)), np.eye(3)]])
    H = np.block([np.eye(3), np.zeros((3,3))])
    q = float(q_acc)
    Q1 = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]], dtype=np.float64) * q
    Q = np.zeros((6,6), dtype=np.float64)
    for a in range(3):
        i, j = a, a+3
        Q[np.ix_([i, j], [i, j])] = Q1
    R = np.eye(3, dtype=np.float64) * float(r_var)

    x = np.zeros(6, dtype=np.float64)
    x[:3] = meas[0]
    if T >= 2:
        x[3:] = meas[1] - meas[0]
    P = np.eye(6, dtype=np.float64) * 100.0

    for t in range(T):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        z = meas[t]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred
        out[t] = x[:3]
    return out


def _kf_ca_positions(meas: np.ndarray, q_jerk: float, r_var: float) -> np.ndarray:
    T = len(meas)
    out = np.full((T, 3), np.nan, dtype=np.float64)
    if T == 0:
        return out
    dt = 1.0
    # state [p,v,a] per axis
    F1 = np.array([[1, dt, 0.5*dt*dt], [0, 1, dt], [0, 0, 1]], dtype=np.float64)
    H1 = np.array([[1, 0, 0]], dtype=np.float64)
    F = np.zeros((9, 9), dtype=np.float64)
    H = np.zeros((3, 9), dtype=np.float64)
    for a in range(3):
        F[a*3:(a+1)*3, a*3:(a+1)*3] = F1
        H[a, a*3:(a+1)*3] = H1

    q = float(q_jerk)
    Q1 = np.array([
        [dt**5/20, dt**4/8, dt**3/6],
        [dt**4/8,  dt**3/3, dt**2/2],
        [dt**3/6,  dt**2/2, dt]
    ], dtype=np.float64) * q
    Q = np.zeros((9, 9), dtype=np.float64)
    for a in range(3):
        Q[a*3:(a+1)*3, a*3:(a+1)*3] = Q1
    R = np.eye(3, dtype=np.float64) * float(r_var)

    x = np.zeros(9, dtype=np.float64)
    x[0] = meas[0, 0]; x[3] = meas[0, 1]; x[6] = meas[0, 2]
    if T >= 2:
        x[1] = meas[1, 0] - meas[0, 0]
        x[4] = meas[1, 1] - meas[0, 1]
        x[7] = meas[1, 2] - meas[0, 2]
    P = np.eye(9, dtype=np.float64) * 100.0

    for t in range(T):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        z = meas[t]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(9) - K @ H) @ P_pred
        out[t] = np.array([x[0], x[3], x[6]], dtype=np.float64)
    return out


def _metric(errors_xyz: np.ndarray) -> Dict[str, object]:
    abs_e = np.abs(errors_xyz)
    mean_abs_dim = abs_e.mean(axis=0)
    var_dim = errors_xyz.var(axis=0)
    rmse_dim = np.sqrt((errors_xyz**2).mean(axis=0))
    p95_dim = np.quantile(abs_e, 0.95, axis=0)
    nrm = np.linalg.norm(errors_xyz, axis=1)
    return {
        'mean_abs_dim': mean_abs_dim.tolist(),
        'var_dim': var_dim.tolist(),
        'rmse_dim': rmse_dim.tolist(),
        'p95_abs_dim': p95_dim.tolist(),
        'mean_abs_norm': float(np.mean(nrm)),
        'var_norm': float(np.var(nrm)),
        'rmse_norm': float(np.sqrt(np.mean(nrm**2))),
        'p95_abs_norm': float(np.quantile(nrm, 0.95)),
    }


def _rebuild_ts_from_sorted(
    series_idx: np.ndarray,
    e_idx: np.ndarray,
    obs_err: np.ndarray,
    cvd_err: np.ndarray,
    cvkf_err: np.ndarray,
    cakf_err: np.ndarray,
    mdl_err: np.ndarray,
) -> Dict[str, Dict[int, List[float]]]:
    order = np.lexsort((e_idx, series_idx))
    ts: Dict[str, Dict[int, List[float]]] = {k: {} for k in ['obs', 'cvd', 'cvkf', 'cakf', 'mdl']}
    arrs = {'obs': obs_err, 'cvd': cvd_err, 'cvkf': cvkf_err, 'cakf': cakf_err, 'mdl': mdl_err}
    for row in order:
        e = int(e_idx[row])
        for k in ts:
            ts[k].setdefault(e, []).append(float(np.linalg.norm(arrs[k][row])))
    return ts


def _finalize_eval_outputs(
    run_dir: Path,
    src_npz: Path,
    use_meas: bool,
    hist: int,
    n_series_total: int,
    obs_err: np.ndarray,
    cvd_err: np.ndarray,
    cvkf_err: np.ndarray,
    cakf_err: np.ndarray,
    mdl_err: np.ndarray,
    ts: Dict[str, Dict[int, List[float]]],
    q_cv: float,
    q_ca: float,
    r_var: float,
    inference: Dict[str, object],
    sharding_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    m_obs = _metric(obs_err)
    m_cvd = _metric(cvd_err)
    m_cvkf = _metric(cvkf_err)
    m_cakf = _metric(cakf_err)
    m_mdl = _metric(mdl_err)
    obs_mean = np.asarray(m_obs['mean_abs_dim'], dtype=np.float64)
    n_windows = int(obs_err.shape[0])
    report: Dict[str, object] = {
        'run_dir': str(run_dir.resolve()),
        'task_mode': 'filter',
        'measurement_conditioning': use_meas,
        'measurement_npz_used': str(src_npz.resolve()),
        'measurement_sigma_m_sidecar': _read_sigma(src_npz),
        'n_test_series': n_series_total,
        'n_windows': n_windows,
        'history_length': hist,
        'methods': {
            'observation': m_obs,
            'cv_diff': m_cvd,
            'cv_kf': m_cvkf,
            'ca_kf': m_cakf,
            'model_filter': m_mdl,
        },
        'denoise_ratio_mean_abs_dim': {
            'model_vs_obs': (np.asarray(m_mdl['mean_abs_dim']) / np.maximum(obs_mean, 1e-12)).tolist(),
            'cv_diff_vs_obs': (np.asarray(m_cvd['mean_abs_dim']) / np.maximum(obs_mean, 1e-12)).tolist(),
            'cv_kf_vs_obs': (np.asarray(m_cvkf['mean_abs_dim']) / np.maximum(obs_mean, 1e-12)).tolist(),
            'ca_kf_vs_obs': (np.asarray(m_cakf['mean_abs_dim']) / np.maximum(obs_mean, 1e-12)).tolist(),
        },
        'denoise_ratio_mean_abs_norm': {
            'model_vs_obs': float(m_mdl['mean_abs_norm'] / max(m_obs['mean_abs_norm'], 1e-12)),
            'cv_diff_vs_obs': float(m_cvd['mean_abs_norm'] / max(m_obs['mean_abs_norm'], 1e-12)),
            'cv_kf_vs_obs': float(m_cvkf['mean_abs_norm'] / max(m_obs['mean_abs_norm'], 1e-12)),
            'ca_kf_vs_obs': float(m_cakf['mean_abs_norm'] / max(m_obs['mean_abs_norm'], 1e-12)),
        },
        'kf_settings': {'cv_q_acc': float(q_cv), 'ca_q_jerk': float(q_ca), 'r_var': float(r_var)},
        'inference': inference,
    }
    if sharding_meta is not None:
        report['sharding'] = sharding_meta

    t_keys = sorted(ts['obs'].keys())
    t = np.asarray(t_keys, dtype=np.int32)
    curves = {}
    for k in ['obs', 'cvd', 'cvkf', 'cakf', 'mdl']:
        curves[k] = np.asarray([np.mean(ts[k][i]) for i in t_keys], dtype=np.float64)

    np.savez_compressed(
        run_dir / 'filter_eval_timeseries.npz',
        time_index=t,
        obs_mean_abs_norm=curves['obs'],
        cv_diff_mean_abs_norm=curves['cvd'],
        cv_kf_mean_abs_norm=curves['cvkf'],
        ca_kf_mean_abs_norm=curves['cakf'],
        model_mean_abs_norm=curves['mdl'],
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
    ax.plot(t, curves['obs'], label='Observation', color='#4C78A8')
    ax.plot(t, curves['cvd'], label='CV-diff', color='#F58518')
    ax.plot(t, curves['cvkf'], label='CV-KF', color='#54A24B')
    ax.plot(t, curves['cakf'], label='CA-KF', color='#B279A2')
    ax.plot(t, curves['mdl'], label='Model-filter', color='#E45756')
    ax.set_xlabel('Time index t')
    ax.set_ylabel('Mean |position error| (m)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(f'{run_dir.name} — error vs time (filter)')
    fig.tight_layout()
    fig.savefig(run_dir / 'filter_eval_timeseries.png', dpi=180)
    plt.close(fig)

    curves_xyz: Dict[str, np.ndarray] = {}
    for name, arr in [
        ('obs', obs_err),
        ('cvd', cvd_err),
        ('cvkf', cvkf_err),
        ('cakf', cakf_err),
        ('mdl', mdl_err),
    ]:
        c = []
        idx = 0
        for e in t_keys:
            n = len(ts['obs'][e])
            c.append(np.mean(np.abs(arr[idx : idx + n, :]), axis=0))
            idx += n
        curves_xyz[name] = np.asarray(c)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ['x', 'y', 'z']
    for d, ax in enumerate(axes):
        ax.plot(t, curves_xyz['obs'][:, d], label='Observation', color='#4C78A8')
        ax.plot(t, curves_xyz['cvd'][:, d], label='CV-diff', color='#F58518')
        ax.plot(t, curves_xyz['cvkf'][:, d], label='CV-KF', color='#54A24B')
        ax.plot(t, curves_xyz['cakf'][:, d], label='CA-KF', color='#B279A2')
        ax.plot(t, curves_xyz['mdl'][:, d], label='Model-filter', color='#E45756')
        ax.set_ylabel(f'Mean |e_{labels[d]}| (m)')
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=8)
    axes[-1].set_xlabel('Time index t')
    fig.suptitle(f'{run_dir.name} — XYZ projected error vs time (filter)')
    fig.tight_layout()
    fig.savefig(run_dir / 'filter_eval_timeseries_xyz.png', dpi=180)
    plt.close(fig)

    report['timeseries_plot_file'] = str((run_dir / 'filter_eval_timeseries.png').resolve())
    report['timeseries_xyz_plot_file'] = str((run_dir / 'filter_eval_timeseries_xyz.png').resolve())
    return report


def merge_filter_eval_shards(run_dir: Path, num_shards: int) -> Dict[str, object]:
    """Merge `filter_eval_shard_{k}_of_{num_shards}.npz` written by sharded evaluate() calls."""
    if num_shards < 2:
        raise ValueError('num_shards must be >= 2 for merge')
    parts = []
    for k in range(num_shards):
        p = run_dir / f'filter_eval_shard_{k}_of_{num_shards}.npz'
        if not p.is_file():
            raise FileNotFoundError(f'missing shard file: {p}')
        parts.append(np.load(p, allow_pickle=True))

    def _cat(key: str) -> np.ndarray:
        return np.concatenate([parts[i][key] for i in range(num_shards)], axis=0)

    series_idx = _cat('series_idx')
    e_idx = _cat('e_idx')
    obs_err = _cat('obs_err')
    cvd_err = _cat('cvd_err')
    cvkf_err = _cat('cvkf_err')
    cakf_err = _cat('cakf_err')
    mdl_err = _cat('mdl_err')

    z0 = parts[0]
    use_meas = bool(int(z0['use_meas']))
    hist = int(z0['hist'])
    n_series_total = int(z0['n_series_total'])
    src_npz = Path(str(z0['measurement_npz_used'].item()))
    q_cv = float(z0['kf_cv_q'])
    q_ca = float(z0['kf_ca_q'])
    r_var = float(z0['kf_r_var'])
    pe = int(z0["predictor_batch_size_effective"])
    inference = {
        "predictor_device": str(z0["predictor_device"].item()),
        "batch_windows": int(z0["batch_windows"]),
        "num_samples": int(z0["num_samples"]),
        "predictor_batch_size_requested": int(z0["predictor_batch_size_requested"]),
        "predictor_batch_size_effective": pe if pe >= 0 else None,
    }
    ts = _rebuild_ts_from_sorted(series_idx, e_idx, obs_err, cvd_err, cvkf_err, cakf_err, mdl_err)
    sharding_meta = {'merged_from_shards': num_shards, 'strategy': 'test_series_modulo'}
    return _finalize_eval_outputs(
        run_dir,
        src_npz,
        use_meas,
        hist,
        n_series_total,
        np.asarray(obs_err, dtype=np.float64),
        np.asarray(cvd_err, dtype=np.float64),
        np.asarray(cvkf_err, dtype=np.float64),
        np.asarray(cakf_err, dtype=np.float64),
        np.asarray(mdl_err, dtype=np.float64),
        ts,
        q_cv,
        q_ca,
        r_var,
        inference,
        sharding_meta,
    )


def evaluate(
    run_dir: Path,
    num_samples: int,
    max_series: int,
    batch_windows: int,
    log_every: int,
    measurement_npz: Optional[Path],
    q_cv: float,
    q_ca: float,
    r_var_override: Optional[float],
    predictor_device: str,
    predictor_batch_size: int,
    num_shards: int = 1,
    shard_id: int = 0,
) -> Dict[str, object]:
    meta = json.loads((run_dir / 'preprocess_metadata.json').read_text(encoding='utf-8'))
    raw = np.load(run_dir / 'train_test_trajectories.npz', allow_pickle=True)
    base_test = [np.asarray(x, dtype=np.float64) for x in raw['base_test_trajectories']]
    test_z = [np.asarray(x, dtype=np.float64) for x in raw['test_trajectories']]
    mean = np.asarray(raw['mean'], dtype=np.float64)
    std = np.asarray(raw['std'], dtype=np.float64)

    use_meas = bool(meta.get('measurement_conditioning', False))
    denorm = bool(meta.get('denorm_metrics', False))
    state_repr = str(meta.get('state_repr', 'absolute'))
    hist = int(meta['history_length'])

    src_npz = measurement_npz if measurement_npz is not None else Path(meta['input_npz'])
    test_idx = [int(i) for i in meta['test_indices']]
    traj_all, meas_all = load_paired_measurements(src_npz)
    traj_t = [traj_all[i] for i in test_idx]
    meas_t = [meas_all[i] for i in test_idx]
    meas_phys = subtract_first_position_from_measurements(meas_t, traj_t, str(meta.get('position_origin', 'none')))

    meas_norm_mean = np.asarray(meta.get('measurement_normalization_mean', []), dtype=np.float64)
    meas_norm_std = np.asarray(meta.get('measurement_normalization_std', []), dtype=np.float64)
    if use_meas and meas_norm_mean.size and meas_norm_std.size:
        meas_z = apply_zscore(meas_phys, meas_norm_mean.astype(np.float32), meas_norm_std.astype(np.float32))
        meas_z = [np.asarray(x, dtype=np.float64) for x in meas_z]
    else:
        meas_z = [m.copy() for m in meas_phys]

    r_var = r_var_override if r_var_override is not None else (float(_read_sigma(src_npz) or 5.0) ** 2)

    from gluonts.model.predictor import Predictor
    from gluonts.dataset.common import ListDataset
    pred_path = run_dir / "predictor"
    if predictor_device == "cpu":
        # Some serialized predictors persist `device: cuda` and ignore
        # deserialize kwargs. Rewrite a temp predictor.json to force CPU.
        with tempfile.TemporaryDirectory(prefix="timegrad_eval_") as td:
            tmp_dir = Path(td) / "predictor"
            shutil.copytree(pred_path, tmp_dir)
            pj = tmp_dir / "predictor.json"
            obj = json.loads(pj.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "kwargs" in obj and isinstance(obj["kwargs"], dict):
                obj["kwargs"]["device"] = "cpu"
            pj.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            orig_torch_load = torch.load
            try:
                def _cpu_load(*args, **kwargs):
                    kwargs["map_location"] = torch.device("cpu")
                    return orig_torch_load(*args, **kwargs)
                torch.load = _cpu_load  # type: ignore[assignment]
                predictor = Predictor.deserialize(tmp_dir)
            finally:
                torch.load = orig_torch_load  # type: ignore[assignment]
    else:
        predictor = Predictor.deserialize(pred_path)
        if predictor_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("predictor_device=cuda but torch.cuda.is_available() is False")
            if hasattr(predictor, "to"):
                predictor.to(torch.device("cuda"))

    if predictor_batch_size > 0 and hasattr(predictor, "batch_size"):
        predictor.batch_size = int(predictor_batch_size)

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if num_shards > 1 and not (0 <= shard_id < num_shards):
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards when num_shards > 1")

    n_series_total = len(base_test) if max_series <= 0 else min(max_series, len(base_test))

    pred_bs_eff = int(getattr(predictor, "batch_size")) if hasattr(predictor, "batch_size") else -1
    inference = {
        "predictor_device": predictor_device,
        "batch_windows": int(batch_windows),
        "num_samples": int(num_samples),
        "predictor_batch_size_requested": int(predictor_batch_size),
        "predictor_batch_size_effective": pred_bs_eff if pred_bs_eff >= 0 else None,
    }

    obs_err: List[np.ndarray] = []
    cvd_err: List[np.ndarray] = []
    cvkf_err: List[np.ndarray] = []
    cakf_err: List[np.ndarray] = []
    mdl_err: List[np.ndarray] = []
    ts = {k: {} for k in ["obs", "cvd", "cvkf", "cakf", "mdl"]}
    col_series_idx: List[int] = []
    col_e_idx: List[int] = []

    pending_items: List[Dict[str, object]] = []
    pending_meta: List[Dict[str, object]] = []
    n_windows = 0
    processed = 0
    prog_prefix = f"[shard {shard_id + 1}/{num_shards}] " if num_shards > 1 else ""

    def flush() -> None:
        nonlocal processed
        if not pending_items:
            return
        ds = ListDataset(pending_items, freq=str(meta["freq"]), one_dim_target=False)
        fcs = list(predictor.predict(ds, num_samples=num_samples))
        for fc, m in zip(fcs, pending_meta):
            e = m["e"]
            base = m["base"]
            truth = m["truth"]
            pred = np.asarray(fc.samples, dtype=np.float64).mean(axis=0)  # (1,D)
            if denorm:
                pred = _denorm(pred, mean, std)
            if state_repr == "delta":
                pred_abs = base[e - 1] + pred[0]
            else:
                pred_abs = pred[0]
            model_pos = pred_abs[:3]

            methods = {
                "obs": m["obs"],
                "cvd": m["cvd"],
                "cvkf": m["cvkf"],
                "cakf": m["cakf"],
                "mdl": model_pos,
            }
            errs = {k: (v - truth) for k, v in methods.items()}
            obs_err.append(errs["obs"])
            cvd_err.append(errs["cvd"])
            cvkf_err.append(errs["cvkf"])
            cakf_err.append(errs["cakf"])
            mdl_err.append(errs["mdl"])
            col_series_idx.append(int(m["series_i"]))
            col_e_idx.append(int(e))
            for k in methods:
                ts[k].setdefault(e, []).append(float(np.linalg.norm(errs[k])))
            processed += 1
            if log_every > 0 and processed % log_every == 0:
                print(f"{prog_prefix}[progress] processed_windows={processed}/{n_windows}", flush=True)
        pending_items.clear()
        pending_meta.clear()

    for i in range(n_series_total):
        if num_shards > 1 and (i % num_shards) != shard_id:
            continue
        base = base_test[i]
        tz = test_z[i]
        mphys = np.asarray(meas_phys[i], dtype=np.float64)
        mz = np.asarray(meas_z[i], dtype=np.float64)
        T = len(base)
        cvkf = _kf_cv_positions(mphys, q_acc=q_cv, r_var=r_var)
        cakf = _kf_ca_positions(mphys, q_jerk=q_ca, r_var=r_var)

        for e in range(max(hist, 2), T):
            truth = base[e, :3]
            obs = mphys[e, :3]
            cvd = _cv_diff_predict(mphys[e - 2, :3], mphys[e - 1, :3])
            cvkf_e = cvkf[e, :3]
            cakf_e = cakf[e, :3]

            item = {"start": datetime(2000, 1, 1), "target": tz[:e, :].T}
            if use_meas:
                item["feat_dynamic_real"] = mz[: e + 1, :].T
            pending_items.append(item)
            pending_meta.append(
                {
                    "e": e,
                    "series_i": i,
                    "base": base,
                    "truth": truth,
                    "obs": obs,
                    "cvd": cvd,
                    "cvkf": cvkf_e,
                    "cakf": cakf_e,
                }
            )
            n_windows += 1
            if len(pending_items) >= batch_windows:
                flush()
    flush()

    obs_a = np.asarray(obs_err, dtype=np.float64)
    cvd_a = np.asarray(cvd_err, dtype=np.float64)
    cvkf_a = np.asarray(cvkf_err, dtype=np.float64)
    cakf_a = np.asarray(cakf_err, dtype=np.float64)
    mdl_a = np.asarray(mdl_err, dtype=np.float64)

    if num_shards > 1:
        shard_path = run_dir / f"filter_eval_shard_{shard_id}_of_{num_shards}.npz"
        W = obs_a.shape[0]
        if W == 0:
            si = np.zeros((0,), dtype=np.int32)
            ei = np.zeros((0,), dtype=np.int32)
            z3 = np.zeros((0, 3), dtype=np.float64)
            obs_a = cvd_a = cvkf_a = cakf_a = mdl_a = z3
        else:
            si = np.asarray(col_series_idx, dtype=np.int32)
            ei = np.asarray(col_e_idx, dtype=np.int32)
        np.savez_compressed(
            shard_path,
            series_idx=si,
            e_idx=ei,
            obs_err=obs_a,
            cvd_err=cvd_a,
            cvkf_err=cvkf_a,
            cakf_err=cakf_a,
            mdl_err=mdl_a,
            use_meas=np.int32(1 if use_meas else 0),
            hist=np.int32(hist),
            n_series_total=np.int32(n_series_total),
            measurement_npz_used=np.array(str(src_npz.resolve())),
            kf_cv_q=np.float64(q_cv),
            kf_ca_q=np.float64(q_ca),
            kf_r_var=np.float64(r_var),
            num_samples=np.int32(num_samples),
            batch_windows=np.int32(batch_windows),
            predictor_batch_size_requested=np.int32(predictor_batch_size),
            predictor_batch_size_effective=np.int32(pred_bs_eff),
            predictor_device=np.array(predictor_device),
        )
        print(f"{prog_prefix}wrote shard {shard_path} windows={W}", flush=True)
        return {
            "sharded": True,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "n_windows_shard": int(W),
            "shard_npz": str(shard_path.resolve()),
        }

    return _finalize_eval_outputs(
        run_dir,
        src_npz,
        use_meas,
        hist,
        n_series_total,
        obs_a,
        cvd_a,
        cvkf_a,
        cakf_a,
        mdl_a,
        ts,
        q_cv,
        q_ca,
        r_var,
        inference,
        None,
    )


def _resolve_predictor_device(s: str) -> str:
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=str, required=True)
    ap.add_argument('--num-samples', type=int, default=100)
    ap.add_argument('--max-series', type=int, default=0)
    ap.add_argument('--batch-windows', type=int, default=1024)
    ap.add_argument('--log-every-windows', type=int, default=1000)
    ap.add_argument('--measurement-npz', type=str, default='')
    ap.add_argument('--kf-cv-q-acc', type=float, default=10.0)
    ap.add_argument('--kf-ca-q-jerk', type=float, default=1.0)
    ap.add_argument('--kf-r-var', type=float, default=-1.0)
    ap.add_argument(
        '--predictor-device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', 'auto'],
        help='GluonTS PyTorch inference device. Use cpu to force CPU (slow). auto picks cuda if available.',
    )
    ap.add_argument(
        '--predictor-batch-size',
        type=int,
        default=512,
        help='Override PyTorchPredictor.batch_size for inference (0 = keep serialized value).',
    )
    ap.add_argument(
        '--num-shards',
        type=int,
        default=1,
        help='Split test series by index %% num_shards (each shard runs on one worker GPU). Merge with --merge-shards.',
    )
    ap.add_argument(
        '--shard-id',
        type=int,
        default=-1,
        help='Which shard this process handles (0 .. num_shards-1). Required when num-shards > 1; ignored when num-shards == 1.',
    )
    ap.add_argument(
        '--merge-shards',
        action='store_true',
        help='Merge filter_eval_shard_*_of_N.npz in run-dir into filter_eval_report.json and plots.',
    )
    ap.add_argument('--out-json', type=str, default='')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.merge_shards:
        if args.num_shards < 2:
            raise SystemExit('--merge-shards requires --num-shards >= 2')
        report = merge_filter_eval_shards(run_dir, args.num_shards)
        out = Path(args.out_json) if args.out_json else (run_dir / 'filter_eval_report.json')
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        m = report['methods']
        print('Merged shard evaluation:')
        print(f"  run_dir={report['run_dir']}")
        print(f"  n_test_series={report['n_test_series']}  n_windows={report['n_windows']}")
        print(f"    model mean|e| (norm)={m['model_filter']['mean_abs_norm']:.6f}")
        print(f"  saved: {out}")
        return

    if args.num_shards > 1 and args.shard_id < 0:
        raise SystemExit('When --num-shards > 1, set --shard-id to 0 .. num-shards-1')

    pred_dev = _resolve_predictor_device(args.predictor_device)
    if pred_dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("predictor-device resolved to cuda but CUDA is not available; use cpu or auto.")

    shard_id = args.shard_id if args.shard_id >= 0 else 0
    report = evaluate(
        run_dir=run_dir,
        num_samples=args.num_samples,
        max_series=args.max_series,
        batch_windows=args.batch_windows,
        log_every=args.log_every_windows,
        measurement_npz=Path(args.measurement_npz) if args.measurement_npz else None,
        q_cv=args.kf_cv_q_acc,
        q_ca=args.kf_ca_q_jerk,
        r_var_override=None if args.kf_r_var < 0 else args.kf_r_var,
        predictor_device=pred_dev,
        predictor_batch_size=args.predictor_batch_size,
        num_shards=args.num_shards,
        shard_id=shard_id,
    )

    if report.get('sharded'):
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    out = Path(args.out_json) if args.out_json else (run_dir / 'filter_eval_report.json')
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    m = report['methods']
    print('Filter evaluation complete:')
    print(f"  run_dir={report['run_dir']}")
    print(f"  n_test_series={report['n_test_series']}  n_windows={report['n_windows']}")
    print('  test-set mean|e| and var(|e| norm):')
    print(f"    observation: mean={m['observation']['mean_abs_norm']:.6f}, var={m['observation']['var_norm']:.6f}")
    print(f"    cv_diff:     mean={m['cv_diff']['mean_abs_norm']:.6f}, var={m['cv_diff']['var_norm']:.6f}")
    print(f"    cv_kf:       mean={m['cv_kf']['mean_abs_norm']:.6f}, var={m['cv_kf']['var_norm']:.6f}")
    print(f"    ca_kf:       mean={m['ca_kf']['mean_abs_norm']:.6f}, var={m['ca_kf']['var_norm']:.6f}")
    print(f"    model:       mean={m['model_filter']['mean_abs_norm']:.6f}, var={m['model_filter']['var_norm']:.6f}")
    print('  mean|e| ratio vs observation (norm):')
    r = report['denoise_ratio_mean_abs_norm']
    print(f"    model_filter: {r['model_vs_obs']:.6f}")
    print(f"    cv_diff:      {r['cv_diff_vs_obs']:.6f}")
    print(f"    cv_kf:        {r['cv_kf_vs_obs']:.6f}")
    print(f"    ca_kf:        {r['ca_kf_vs_obs']:.6f}")
    print(f"  saved: {out}")


if __name__ == '__main__':
    main()
