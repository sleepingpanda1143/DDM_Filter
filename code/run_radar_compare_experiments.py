#!/usr/bin/env python3
"""
Run a fixed matrix of radar TimeGrad experiments and aggregate metrics.

Usage (from repo root or this directory):
  PYTHONPATH=/path/to/DDM_Timeseries_Forecast python run_radar_compare_experiments.py

Requires: torch, gluonts, and DDM_Timeseries_Forecast deps (see setup.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
RADAR = ROOT / "Radar"
DDM = ROOT / "DDM_Timeseries_Forecast"
CODE = DDM / "code"
DEFAULT_NPZ = RADAR / "data" / "trajectory_data.npz"
LOW_NPZ = RADAR / "data" / "trajectory_low_scale_data.npz"
GEN_SCRIPT = RADAR / "generate_trajectory_data.py"
DEFAULT_OUT_ROOT = DDM / "dataset" / "radar_compare_runs"


@dataclass
class Experiment:
    name: str
    description: str
    extra_args: List[str] = field(default_factory=list)
    input_npz: Optional[Path] = None
    generate_low_scale_first: bool = False


def build_experiments() -> List[Experiment]:
    """Ordered comparison matrix (keep names stable for the report)."""
    common_long = [
        "--epochs",
        "15",
        "--num-batches-per-epoch",
        "60",
        "--batch-size",
        "32",
        "--max-train-trajectories",
        "0",
        "--max-test-trajectories",
        "0",
        "--num-samples",
        "50",
        "--denorm-metrics",
        "--suppress-warnings",
        "--save-train-loss",
    ]
    return [
        Experiment(
            name="01_baseline_short",
            description="Original-style budget: 2 epochs × 20 batches, cap 300 train / 80 test, TimeGrad MeanScaler on.",
            extra_args=[
                "--epochs",
                "2",
                "--num-batches-per-epoch",
                "20",
                "--max-train-trajectories",
                "300",
                "--max-test-trajectories",
                "80",
                "--num-samples",
                "50",
                "--denorm-metrics",
                "--suppress-warnings",
            ],
        ),
        Experiment(
            name="02_long_train_double_scale",
            description="More optimization steps; full train/test split; z-score + TimeGrad MeanScaler (double scaling).",
            extra_args=common_long,
        ),
        Experiment(
            name="03_long_train_zscore_only",
            description="Same as 02 but --no-timegrad-scaling (only dataset z-score).",
            extra_args=common_long + ["--no-timegrad-scaling"],
        ),
        Experiment(
            name="04_long_train_wider_rnn",
            description="Same as 03 with larger GRU (num_cells=96, num_layers=2).",
            extra_args=common_long
            + [
                "--no-timegrad-scaling",
                "--num-cells",
                "96",
                "--num-layers",
                "2",
            ],
        ),
        Experiment(
            name="05_low_distance_speed_npz",
            description="Low distance/speed synthetic data (trajectory_low_scale.npz) + same training as 03.",
            extra_args=common_long + ["--no-timegrad-scaling"],
            input_npz=LOW_NPZ,
            generate_low_scale_first=True,
        ),
    ]


def ensure_low_scale_npz(path: Path, python_exe: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(GEN_SCRIPT),
        "--num_trajectories",
        "2000",
        "--min_length",
        "80",
        "--max_length",
        "220",
        "--output_dir",
        str(path.parent),
        "--prefix",
        "trajectory_low_scale",
        "--seed",
        "43",
        "--init-x-range",
        "-2500",
        "2500",
        "--init-y-range",
        "-2500",
        "2500",
        "--cv-speed-range",
        "5",
        "55",
    ]
    print("Generating low-scale dataset:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(RADAR))


def run_one(
    python_exe: str,
    exp: Experiment,
    default_npz: Path,
    device: str,
    freq: str,
    out_root: Path,
) -> Dict[str, Any]:
    input_npz = exp.input_npz or default_npz
    if exp.generate_low_scale_first:
        ensure_low_scale_npz(LOW_NPZ, python_exe)

    out_dir = out_root / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(CODE / "radar_timegrad_experiment.py"),
        "--input-npz",
        str(input_npz),
        "--output-dir",
        str(out_dir),
        "--device",
        device,
        "--freq",
        freq,
        *exp.extra_args,
    ]

    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(DDM) + __import__("os").pathsep + env.get("PYTHONPATH", "")

    print("\n===", exp.name, "===\n", " ".join(cmd), "\n", flush=True)
    proc = subprocess.run(cmd, cwd=str(CODE), env=env, capture_output=True, text=True)
    log_path = out_dir / "run.log"
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {
            "name": exp.name,
            "description": exp.description,
            "ok": False,
            "returncode": proc.returncode,
            "log": str(log_path),
            "metrics": None,
        }

    metrics_path = out_dir / "metrics.json"
    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    row: Dict[str, Any] = {
        "name": exp.name,
        "description": exp.description,
        "ok": True,
        "output_dir": str(out_dir),
        "input_npz": str(input_npz),
        "metrics": metrics,
    }
    return row


def write_report(rows: List[Dict[str, Any]], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Radar TimeGrad comparison runs\n")
    lines.append(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("\n## Data setup (training source)\n")
    lines.append(
        "- **NPZ format**: `trajectories` object array; each array is `(T, 4)` with columns `[x, y, vx, vy]` in meters and m/s.\n"
    )
    lines.append(
        "- **Generator**: `Radar/generate_trajectory_data.py` (CV / CA / CT mixture unless `--cv-only`). "
        "Not StoneSoup CSV; StoneSoup benchmark script is separate.\n"
    )
    lines.append(
        "- **Labels**: each timestep is **ground-truth state**; the model predicts the **next** window from **past** context only (no explicit measurement noise in the series), i.e. **GT history → predict GT future**.\n"
    )
    lines.append("\n## Metrics table\n")
    lines.append(
        "| Run | ADE (m) | RMSE_pos | CV_ADE | Model/CV ADE | Notes |\n"
        "|-----|---------|----------|--------|--------------|-------|\n"
    )
    for row in rows:
        if not row.get("ok"):
            lines.append(f"| {row['name']} | — | — | — | — | FAILED (see run.log) |\n")
            continue
        m = row.get("metrics") or {}
        lines.append(
            f"| {row['name']} | {m.get('ADE', 'n/a')} | {m.get('RMSE_pos', 'n/a')} | "
            f"{m.get('CV_ADE', 'n/a')} | {m.get('Model_vs_CV_ADE_Ratio', 'n/a')} | {row.get('description', '')[:60]}… |\n"
        )

    lines.append("\n## Per-run details\n")
    for row in rows:
        lines.append(f"### {row['name']}\n")
        lines.append(f"- {row.get('description', '')}\n")
        lines.append(f"- Output: `{row.get('output_dir', '')}`\n")
        if row.get("metrics"):
            lines.append("\n```json\n")
            lines.append(json.dumps(row["metrics"], indent=2))
            lines.append("\n```\n")

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=str, default=sys.executable, help="Python to use.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--freq",
        type=str,
        default="1min",
        help="GluonTS frequency string (default matches previous experiment).",
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        default=str(DEFAULT_NPZ),
        help="Default trajectory npz for runs that do not override it.",
    )
    parser.add_argument("--skip-generate", action="store_true", help="Do not create low-scale npz if missing.")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated experiment names (e.g. 01_baseline_short,03_long_train_zscore_only). Empty = run all.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
        help="Directory for per-experiment outputs, comparison_summary.json, and comparison_report.md.",
    )
    args = parser.parse_args()
    out_root = Path(args.out_root)

    default_npz = Path(args.input_npz)
    if not default_npz.exists():
        print(f"Missing default npz: {default_npz}", file=sys.stderr)
        sys.exit(1)

    experiments = build_experiments()
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        by_name = {e.name: e for e in experiments}
        missing_names = wanted - by_name.keys()
        if missing_names:
            print(f"Unknown --only names: {sorted(missing_names)}", file=sys.stderr)
            print(f"Valid names: {sorted(by_name.keys())}", file=sys.stderr)
            sys.exit(1)
        experiments = [e for e in experiments if e.name in wanted]
        if not experiments:
            print("No experiments selected after --only filter.", file=sys.stderr)
            sys.exit(1)
    if args.skip_generate:
        for e in experiments:
            if e.generate_low_scale_first:
                e.generate_low_scale_first = False

    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        rows.append(
            run_one(
                args.python,
                exp,
                default_npz=default_npz,
                device=args.device,
                freq=args.freq,
                out_root=out_root,
            )
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "comparison_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(rows, out_root / "comparison_report.md")
    print(f"\nWrote {summary_path} and {out_root / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
