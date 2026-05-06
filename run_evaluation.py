"""Run evaluation on a submission HDF5 file against the ground truth HDF5 file."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.evals.evals import evaluate_submission
from src.evals.eval_utils import evaluation_results_to_dataframe


# Main function to run the evaluation script
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-h5", type=Path, help="Path to submission HDF5 file")
    parser.add_argument("--truth-h5", type=Path, help="Path to ground truth HDF5 file")
    parser.add_argument("--config-dir", type=Path, help="Path to dataset config directory")
    parser.add_argument(
        "--experiment-type",
        type=str,
        required=True,
        choices=("memory_network", "pass_decision", "multi_task"),
        help="Which benchmark the HDF5 files belong to.",
    )
    parser.add_argument(
        "--output-dist",
        type=str,
        required=True,
        choices=("gaussian", "poisson"),
        help="How neural activity is represented: Gaussian (rates / continuous) vs Poisson (counts).",
    )
    parser.add_argument(
        "--truth-time-start",
        type=int,
        default=0,
        metavar="k",
        help=(
            "Truth time index that aligns with prediction time 0. Use 0 when predictions cover the full truth window."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default="evaluation_results.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    submission_h5 = args.submission_h5.expanduser().resolve()
    truth_h5 = args.truth_h5.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    model_yaml = config_dir / "model" / "model.yaml"

    if not submission_h5.is_file():
        raise SystemExit(f"Submission HDF5 not found: {submission_h5}")
    if not truth_h5.is_file():
        raise SystemExit(f"Truth HDF5 not found: {truth_h5}")
    if not config_dir.is_dir():
        raise SystemExit(f"Config directory not found: {config_dir}")
    if not model_yaml.is_file():
        raise SystemExit(f"Model config not found: {model_yaml}")

    results = evaluate_submission(
        submission_h5=submission_h5,
        truth_h5=truth_h5,
        config_dir=config_dir,
        experiment_type=args.experiment_type,
        output_dist=args.output_dist,
        truth_time_start=args.truth_time_start,
    )

    dataframe = evaluation_results_to_dataframe(results)
    out_csv = args.output_csv.expanduser().resolve()
    dataframe.to_csv(out_csv, index=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(dataframe.to_string(index=False))
    print(f"Wrote to {out_csv}")

if __name__ == "__main__":
    main()