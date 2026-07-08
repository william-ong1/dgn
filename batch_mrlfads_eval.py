#!/usr/bin/env python3
"""
Run MR-LFADS forward + evaluation for every checkpoint run in a directory.

For each run:
  1. mrlfads_forward_h5.py  -> model_outputs/<run_name>_outputs.h5
  2. run_evaluation.py      -> model_outputs/<run_name>_eval.csv

Then writes a combined summary CSV (one row per run, global metrics).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evals.evals import evaluate_submission
from src.evals.eval_utils import split_results_for_display


def _discover_runs(runs_dir: Path, run_glob: str | None = None) -> list[Path]:
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    if run_glob:
        candidates = [p for p in candidates if p.match(run_glob)]

    runs = []
    for run_dir in candidates:
        if not (run_dir / "configs" / "main.yaml").is_file():
            continue
        ckpt_dir = run_dir / "lightning_checkpoints"
        if not ckpt_dir.is_dir() or not any(ckpt_dir.glob("*.ckpt")):
            print(f"skip {run_dir.name}: no checkpoint")
            continue
        runs.append(run_dir)
    return runs


def _run_forward(
    repo_root: Path,
    run_dir: Path,
    input_h5: Path,
    output_h5: Path,
    output_dist: str,
    accelerator: str,
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "mrlfads_forward_h5.py"),
        "--run-dir",
        str(run_dir),
        "--input-h5",
        str(input_h5),
        "--output-h5",
        str(output_h5),
        "--output-dist",
        output_dist,
        "--accelerator",
        accelerator,
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)


def _run_eval(
    submission_h5: Path,
    truth_h5: Path,
    config_dir: Path,
    experiment_type: str,
    output_dist: str,
    truth_time_start: int,
    output_csv: Path,
    rates_truth_h5: Path | None = None,
    poisson_dt: float = 0.01,
    poisson_rate_max: float = 40.0,
) -> dict:
    results = evaluate_submission(
        submission_h5=submission_h5,
        truth_h5=truth_h5,
        config_dir=config_dir,
        experiment_type=experiment_type,
        output_dist=output_dist,
        truth_time_start=truth_time_start,
        rates_truth_h5=rates_truth_h5,
        poisson_dt=poisson_dt,
        poisson_rate_max=poisson_rate_max,
    )
    region_df, global_df, temporal_df = split_results_for_display(results)

    region_csv = region_df.copy()
    if not region_csv.empty:
        region_csv.insert(0, "section", "region")
    global_csv = global_df.copy()
    if not global_csv.empty:
        global_csv.insert(0, "section", "global")
    temporal_csv = temporal_df.copy()
    if not temporal_csv.empty:
        temporal_csv.insert(0, "section", "temporal")

    out = pd.concat([region_csv, temporal_csv, global_csv], ignore_index=True, sort=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return results


def _global_row(run_name: str, results: dict) -> dict:
    _, global_df, temporal_df = split_results_for_display(results)
    row = {"run": run_name}
    if global_df.empty:
        pass
    else:
        for col in global_df.columns:
            row[col] = global_df.iloc[0][col]
    if not temporal_df.empty:
        for col in temporal_df.columns:
            row[f"temporal-{col}"] = temporal_df.iloc[0][col]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing MR-LFADS run folders (each with configs/ and lightning_checkpoints/).",
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="Observed activity HDF5 passed to mrlfads_forward_h5.py.",
    )
    parser.add_argument(
        "--truth-h5",
        type=Path,
        required=True,
        help="Ground-truth HDF5 for run_evaluation.py (often same as --input-h5).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/memory_network"),
        help="Evaluation config directory (e.g. configs/memory_network/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_outputs"),
        help="Where to write <run>_outputs.h5 and <run>_eval.csv.",
    )
    parser.add_argument(
        "--experiment-type",
        default="memory_network",
        choices=("memory_network", "pass_decision", "multi_task"),
    )
    parser.add_argument(
        "--output-dist",
        default="poisson",
        choices=("gaussian", "poisson"),
    )
    parser.add_argument(
        "--truth-time-start",
        type=int,
        default=10,
        help="Align prediction time 0 to truth time k (ic_enc_seq_len for memory network).",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        help="PyTorch Lightning accelerator for forward pass (gpu, cuda, mps, cpu).",
    )
    parser.add_argument(
        "--run-glob",
        default=None,
        help="Optional glob to filter run folder names (e.g. 'mn_pois_*').",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip forward if <run>_outputs.h5 already exists.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Only run mrlfads_forward_h5.py (no evaluation).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing output H5 files.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("model_outputs/evaluation_summary_all_runs.csv"),
        help="Combined CSV with one row per run (global metrics).",
    )
    parser.add_argument(
        "--rates-truth-h5",
        type=Path,
        default=None,
        help="Optional gaussian.h5 for rates-vs-rates R² (continuous activity -> λ).",
    )
    parser.add_argument(
        "--poisson-dt",
        type=float,
        default=0.01,
        help="Bin width (s) for converting continuous activity to λ.",
    )
    parser.add_argument(
        "--poisson-rate-max",
        type=float,
        default=40.0,
        help="Max firing rate (Hz) for converting continuous activity to λ.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = args.runs_dir.expanduser().resolve()
    input_h5 = args.input_h5.expanduser().resolve()
    truth_h5 = args.truth_h5.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve()

    if not input_h5.is_file():
        raise SystemExit(f"input H5 not found: {input_h5}")
    if not truth_h5.is_file():
        raise SystemExit(f"truth H5 not found: {truth_h5}")
    if not config_dir.is_dir():
        raise SystemExit(f"config dir not found: {config_dir}")

    runs = _discover_runs(runs_dir, run_glob=args.run_glob)
    if not runs:
        raise SystemExit(f"No runnable checkpoints found in {runs_dir}")

    summary_rows: list[dict] = []
    region_rows: list[pd.DataFrame] = []

    for run_dir in runs:
        run_name = run_dir.name
        output_h5 = output_dir / f"{run_name}_outputs.h5"
        eval_csv = output_dir / f"{run_name}_eval.csv"

        if not args.eval_only:
            if args.skip_existing and output_h5.is_file():
                print(f"skip forward {run_name}: {output_h5} exists")
            else:
                _run_forward(
                    repo_root=repo_root,
                    run_dir=run_dir,
                    input_h5=input_h5,
                    output_h5=output_h5,
                    output_dist=args.output_dist,
                    accelerator=args.accelerator,
                )

        if args.forward_only:
            continue

        if not output_h5.is_file():
            print(f"skip eval {run_name}: missing {output_h5}")
            continue

        print(f">> evaluate {run_name}")
        results = _run_eval(
            submission_h5=output_h5,
            truth_h5=truth_h5,
            config_dir=config_dir,
            experiment_type=args.experiment_type,
            output_dist=args.output_dist,
            truth_time_start=args.truth_time_start,
            output_csv=eval_csv,
            rates_truth_h5=args.rates_truth_h5,
            poisson_dt=args.poisson_dt,
            poisson_rate_max=args.poisson_rate_max,
        )
        summary_rows.append(_global_row(run_name, results))

        region_df, _, _ = split_results_for_display(results)
        if not region_df.empty:
            r = region_df.copy()
            r.insert(0, "run", run_name)
            region_rows.append(r)

        output_h5.unlink()
        print(f"deleted forward output: {output_h5}")

    if args.forward_only:
        return

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nWrote summary: {summary_csv}")
        print(summary_df.to_string(index=False))

    if region_rows:
        region_path = summary_csv.with_name(summary_csv.stem + "_by_region.csv")
        pd.concat(region_rows, ignore_index=True).to_csv(region_path, index=False)
        print(f"Wrote per-region: {region_path}")


if __name__ == "__main__":
    main()
