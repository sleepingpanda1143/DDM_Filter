# Radar TimeGrad comparison runs
Generated (UTC): 2026-03-29T14:45:20.598251+00:00

## Data setup (training source)
- **NPZ format**: `trajectories` object array; each array is `(T, 4)` with columns `[x, y, vx, vy]` in meters and m/s.
- **Generator**: `Radar/generate_trajectory_data.py` (CV / CA / CT mixture unless `--cv-only`). Not StoneSoup CSV; StoneSoup benchmark script is separate.
- **Labels**: each timestep is **ground-truth state**; the model predicts the **next** window from **past** context only (no explicit measurement noise in the series), i.e. **GT history → predict GT future**.

## Metrics table
| Run | ADE (m) | RMSE_pos | CV_ADE | Model/CV ADE | Notes |
|-----|---------|----------|--------|--------------|-------|
| 04_long_train_wider_rnn | 9662.771484375 | 8746.7919921875 | 2.5136497020721436 | 3844.120155807482 | Same as 03 with larger GRU (num_cells=96, num_layers=2).… |
| 05_low_distance_speed_npz | 2692.824462890625 | 2851.10791015625 | 1.218184232711792 | 2210.5231627371713 | Low distance/speed synthetic data (trajectory_low_scale.npz)… |

## Per-run details
### 04_long_train_wider_rnn
- Same as 03 with larger GRU (num_cells=96, num_layers=2).
- Output: `/home/xiongmaoren/kalman_net/DDM_Timeseries_Forecast/dataset/radar_compare_runs_gpu1/04_long_train_wider_rnn`

```json
{
  "CRPS": 0.24988901112060544,
  "ND": 0.3209581969843166,
  "NRMSE": 0.45164355835489883,
  "CRPS_Sum": 0.08662037682522528,
  "ND_Sum": 0.11009356613627871,
  "NRMSE_Sum": 0.1674422900373248,
  "ADE": 9662.771484375,
  "FDE": 9662.771484375,
  "RMSE_pos": 8746.7919921875,
  "RMSE_vel": 104.3178482055664,
  "CV_ADE": 2.5136497020721436,
  "CV_FDE": 2.5136497020721436,
  "CV_RMSE_pos": 3.197056770324707,
  "CV_RMSE_vel": 6.345277309417725,
  "Model_vs_CV_ADE_Ratio": 3844.120155807482,
  "Model_vs_CV_RMSE_pos_Ratio": 2735.888856705894
}
```
### 05_low_distance_speed_npz
- Low distance/speed synthetic data (trajectory_low_scale.npz) + same training as 03.
- Output: `/home/xiongmaoren/kalman_net/DDM_Timeseries_Forecast/dataset/radar_compare_runs_gpu1/05_low_distance_speed_npz`

```json
{
  "CRPS": 0.2896647688022643,
  "ND": 0.3692371176074004,
  "NRMSE": 0.5993009900883854,
  "CRPS_Sum": 0.15607965074801886,
  "ND_Sum": 0.19491847387234496,
  "NRMSE_Sum": 0.33936262445963494,
  "ADE": 2692.824462890625,
  "FDE": 2692.824462890625,
  "RMSE_pos": 2851.10791015625,
  "RMSE_vel": 55.992454528808594,
  "CV_ADE": 1.218184232711792,
  "CV_FDE": 1.218184232711792,
  "CV_RMSE_pos": 1.389183521270752,
  "CV_RMSE_vel": 2.723538875579834,
  "Model_vs_CV_ADE_Ratio": 2210.5231627371713,
  "Model_vs_CV_RMSE_pos_Ratio": 2052.3623167860546
}
```
