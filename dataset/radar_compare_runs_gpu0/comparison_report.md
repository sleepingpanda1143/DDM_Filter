# Radar TimeGrad comparison runs
Generated (UTC): 2026-03-29T14:45:01.972078+00:00

## Data setup (training source)
- **NPZ format**: `trajectories` object array; each array is `(T, 4)` with columns `[x, y, vx, vy]` in meters and m/s.
- **Generator**: `Radar/generate_trajectory_data.py` (CV / CA / CT mixture unless `--cv-only`). Not StoneSoup CSV; StoneSoup benchmark script is separate.
- **Labels**: each timestep is **ground-truth state**; the model predicts the **next** window from **past** context only (no explicit measurement noise in the series), i.e. **GT history → predict GT future**.

## Metrics table
| Run | ADE (m) | RMSE_pos | CV_ADE | Model/CV ADE | Notes |
|-----|---------|----------|--------|--------------|-------|
| 01_baseline_short | 31362.90234375 | 29968.236328125 | 1.9712051153182983 | 15910.52199490955 | Original-style budget: 2 epochs × 20 batches, cap 300 train … |
| 02_long_train_double_scale | 2329.089111328125 | 2852.96435546875 | 2.5136497020721436 | 926.576646462761 | More optimization steps; full train/test split; z-score + Ti… |
| 03_long_train_zscore_only | 9994.2685546875 | 8953.5654296875 | 2.5136497020721436 | 3975.998941478862 | Same as 02 but --no-timegrad-scaling (only dataset z-score).… |

## Per-run details
### 01_baseline_short
- Original-style budget: 2 epochs × 20 batches, cap 300 train / 80 test, TimeGrad MeanScaler on.
- Output: `/home/xiongmaoren/kalman_net/DDM_Timeseries_Forecast/dataset/radar_compare_runs_gpu0/01_baseline_short`

```json
{
  "CRPS": 0.5063892754168826,
  "ND": 0.715834942915428,
  "NRMSE": 1.0972477146576824,
  "CRPS_Sum": 0.7104990844624574,
  "ND_Sum": 0.9110217170749649,
  "NRMSE_Sum": 1.3541143702400622,
  "ADE": 31362.90234375,
  "FDE": 31362.90234375,
  "RMSE_pos": 29968.236328125,
  "RMSE_vel": 187.0576171875,
  "CV_ADE": 1.9712051153182983,
  "CV_FDE": 1.9712051153182983,
  "CV_RMSE_pos": 2.360452890396118,
  "CV_RMSE_vel": 4.684606552124023,
  "Model_vs_CV_ADE_Ratio": 15910.52199490955,
  "Model_vs_CV_RMSE_pos_Ratio": 12695.968832953873
}
```
### 02_long_train_double_scale
- More optimization steps; full train/test split; z-score + TimeGrad MeanScaler (double scaling).
- Output: `/home/xiongmaoren/kalman_net/DDM_Timeseries_Forecast/dataset/radar_compare_runs_gpu0/02_long_train_double_scale`

```json
{
  "CRPS": 0.07086691770949828,
  "ND": 0.08690197703667739,
  "NRMSE": 0.1604404273449177,
  "CRPS_Sum": 0.05607805438201063,
  "ND_Sum": 0.07072795278117167,
  "NRMSE_Sum": 0.12576548891627806,
  "ADE": 2329.089111328125,
  "FDE": 2329.089111328125,
  "RMSE_pos": 2852.96435546875,
  "RMSE_vel": 38.89090347290039,
  "CV_ADE": 2.5136497020721436,
  "CV_FDE": 2.5136497020721436,
  "CV_RMSE_pos": 3.197056770324707,
  "CV_RMSE_vel": 6.345277309417725,
  "Model_vs_CV_ADE_Ratio": 926.576646462761,
  "Model_vs_CV_RMSE_pos_Ratio": 892.3721286247259
}
```
### 03_long_train_zscore_only
- Same as 02 but --no-timegrad-scaling (only dataset z-score).
- Output: `/home/xiongmaoren/kalman_net/DDM_Timeseries_Forecast/dataset/radar_compare_runs_gpu0/03_long_train_zscore_only`

```json
{
  "CRPS": 0.2509160490864641,
  "ND": 0.32124026925117516,
  "NRMSE": 0.45105757691187226,
  "CRPS_Sum": 0.08743300773695484,
  "ND_Sum": 0.11156525694990094,
  "NRMSE_Sum": 0.17120075310136157,
  "ADE": 9994.2685546875,
  "FDE": 9994.2685546875,
  "RMSE_pos": 8953.5654296875,
  "RMSE_vel": 102.7352066040039,
  "CV_ADE": 2.5136497020721436,
  "CV_FDE": 2.5136497020721436,
  "CV_RMSE_pos": 3.197056770324707,
  "CV_RMSE_vel": 6.345277309417725,
  "Model_vs_CV_ADE_Ratio": 3975.998941478862,
  "Model_vs_CV_RMSE_pos_Ratio": 2800.565042446255
}
```
