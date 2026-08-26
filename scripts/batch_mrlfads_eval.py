#!/usr/bin/env python3
"""
Run MR-LFADS forward + evaluation for every checkpoint run in a directory.

For each run:
  1. mrlfads_forward_h5.py  -> model_outputs/<run_name>_outputs.h5
  2. run_evaluation.py      -> model_outputs/<run_name>_eval.csv

Then writes a combined summary CSV with:
  - one row per run × region (e.g. A0, A1, …)
  - plus a ``region=combined`` row when multiple areas are present

Multi-task (mt_pois) example::

    python scripts/batch_mrlfads_eval.py \\
        --runs-dir /path/to/mrlfads/mt_runs \\
        --experiment-type multi_task \\
        --output-dist poisson \\
        --recursive \\
        --resolve-data-from-run

Per-run ``data.h5`` and ``ic_enc_seq_len`` are read from each run's saved configs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# scripts/ -> repo root (needed for ``from src.evals...``)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evals.evals import evaluate_submission
from src.evals.eval_utils import (
    discover_mrlfads_runs,
    load_mrlfads_ic_enc_seq_len,
    resolve_mrlfads_run_data_h5,
    results_to_summary_rows,
    split_results_for_display,
)


def _discover_runs(
    runs_dir: Path,
    run_glob: str | None = None,
    *,
    recursive: bool = False,
) -> list[Path]:
    return discover_mrlfads_runs(runs_dir, run_glob=run_glob, recursive=recursive)


def _resolve_truth_time_start(
    *,
    run_dir: Path,
    experiment_type: str,
    truth_time_start: int | str | None,
) -> int:
    if truth_time_start is None or str(truth_time_start).lower() == "auto":
        if experiment_type in ("memory_network", "multi_task"):
            return load_mrlfads_ic_enc_seq_len(run_dir)
        return 0
    return int(truth_time_start)


def _resolve_run_input_h5(
    run_dir: Path,
    *,
    input_h5: Path | None,
    resolve_data_from_run: bool,
) -> Path:
    if input_h5 is not None:
        return input_h5
    if not resolve_data_from_run:
        raise ValueError(
            f"No --input-h5 for run {run_dir.name}; pass --input-h5 or --resolve-data-from-run."
        )
    return resolve_mrlfads_run_data_h5(run_dir)


def _run_forward(
    repo_root: Path,
    run_dir: Path,
    input_h5: Path,
    output_h5: Path,
    output_dist: str,
    accelerator: str,
    experiment_type: str,
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "mrlfads_forward_h5.py"),
        "--run-dir",
        str(run_dir),
        "--input-h5",
        str(input_h5),
        "--output-h5",
        str(output_h5),
        "--output-dist",
        output_dist,
        "--experiment-type",
        experiment_type,
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
    """Deprecated helper kept for callers; prefer ``results_to_summary_rows``."""
    rows = results_to_summary_rows(run_name, results)
    combined = next((r for r in rows if r.get("region") == "combined"), None)
    return combined if combined is not None else (rows[0] if rows else {"run": run_name})


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
        default=None,
        help=(
            "Observed activity HDF5 passed to mrlfads_forward_h5.py (must be named data.h5). "
            "Omit when using --resolve-data-from-run to read each run's datamodule config."
        ),
    )
    parser.add_argument(
        "--truth-h5",
        type=Path,
        default=None,
        help=(
            "Ground-truth HDF5 for evaluation (often same as --input-h5). "
            "Defaults to the per-run input H5 when omitted."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation config directory (e.g. configs/memory_network/ or configs/multi_task/). "
            "Defaults to configs/<experiment-type>/ under the repo root."
        ),
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
        default="auto",
        help=(
            "Align prediction time 0 to truth time k. Use 'auto' to read ic_enc_seq_len "
            "from each run's model config (memory_network / multi_task), or 0 for pass_decision."
        ),
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        help="PyTorch Lightning accelerator for forward pass (gpu, cuda, mps, cpu).",
    )
    parser.add_argument(
        "--run-glob",
        default=None,
        help="Optional glob to filter run folder names (e.g. 'rt_go_kl*' or '*_kl0001_*').",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search nested subdirectories for run folders (e.g. rt_go/<run>/).",
    )
    parser.add_argument(
        "--resolve-data-from-run",
        action="store_true",
        help=(
            "Read each run's training data.h5 from configs/datamodule/datamodule.yaml "
            "(datapath_override). Default for multi_task when --input-h5 is omitted."
        ),
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
        help=(
            "Combined CSV: one row per run×region, plus region=combined when "
            "multiple areas are present."
        ),
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

    repo_root = REPO_ROOT
    runs_dir = args.runs_dir.expanduser().resolve()

    resolve_data_from_run = args.resolve_data_from_run or (
        args.input_h5 is None and args.experiment_type == "multi_task"
    )
    input_h5_global = (
        args.input_h5.expanduser().resolve() if args.input_h5 is not None else None
    )
    truth_h5_global = (
        args.truth_h5.expanduser().resolve() if args.truth_h5 is not None else None
    )

    config_dir = args.config_dir
    if config_dir is None:
        config_dir = Path("configs") / args.experiment_type
    config_dir = config_dir.expanduser()
    config_dir = (
        config_dir.resolve()
        if config_dir.is_absolute()
        else (repo_root / config_dir).resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve()
    rates_truth_h5 = (
        args.rates_truth_h5.expanduser().resolve()
        if args.rates_truth_h5 is not None
        else None
    )

    if input_h5_global is not None and input_h5_global.name != "data.h5":
        raise SystemExit(
            f"--input-h5 must be named data.h5 (got {input_h5_global.name}); "
            "mrlfads_forward_h5 sets datapath_override to its parent directory."
        )
    if input_h5_global is not None and not input_h5_global.is_file():
        raise SystemExit(f"input H5 not found: {input_h5_global}")
    if not resolve_data_from_run and input_h5_global is None:
        raise SystemExit("Pass --input-h5 or --resolve-data-from-run.")
    if truth_h5_global is not None and not truth_h5_global.is_file():
        raise SystemExit(f"truth H5 not found: {truth_h5_global}")
    if not config_dir.is_dir():
        raise SystemExit(f"config dir not found: {config_dir}")

    runs = _discover_runs(runs_dir, run_glob=args.run_glob, recursive=args.recursive)
    if not runs:
        raise SystemExit(f"No runnable checkpoints found in {runs_dir}")

    summary_rows: list[dict] = []
    region_rows: list[pd.DataFrame] = []

    for run_dir in runs:
        run_name = run_dir.name
        output_h5 = output_dir / f"{run_name}_outputs.h5"
        eval_csv = output_dir / f"{run_name}_eval.csv"

        try:
            input_h5 = _resolve_run_input_h5(
                run_dir,
                input_h5=input_h5_global,
                resolve_data_from_run=resolve_data_from_run,
            )
            truth_h5 = truth_h5_global if truth_h5_global is not None else input_h5
            truth_time_start = _resolve_truth_time_start(
                run_dir=run_dir,
                experiment_type=args.experiment_type,
                truth_time_start=args.truth_time_start,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"skip {run_name}: {e}")
            continue

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
                    experiment_type=args.experiment_type,
                )

        if args.forward_only:
            continue

        if not output_h5.is_file():
            print(f"skip eval {run_name}: missing {output_h5}")
            continue

        print(f">> evaluate {run_name} (truth_time_start={truth_time_start}, data={input_h5})")
        try:
            results = _run_eval(
                submission_h5=output_h5,
                truth_h5=truth_h5,
                config_dir=config_dir,
                experiment_type=args.experiment_type,
                output_dist=args.output_dist,
                truth_time_start=truth_time_start,
                output_csv=eval_csv,
                rates_truth_h5=rates_truth_h5,
                poisson_dt=args.poisson_dt,
                poisson_rate_max=args.poisson_rate_max,
            )
        except Exception as e:
            print(f"skip eval {run_name}: {type(e).__name__}: {e}")
            continue

        summary_rows.extend(results_to_summary_rows(run_name, results))

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
