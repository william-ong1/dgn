"""Run evaluation on a submission HDF5 file against the ground truth HDF5 file."""

import argparse
from pathlib import Path
from src.evals.evals import evaluate_submission

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
        "--distribution",
        type=str,
        required=True,
        choices=("gaussian", "poisson"),
        help="How activity is represented: Gaussian (rates / continuous) vs Poisson (counts).",
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
    args = parser.parse_args()

    results = evaluate_submission(
            submission_h5=args.submission_h5,
            truth_h5=args.truth_h5,
            config_dir=args.config_dir,
            experiment_type=args.experiment_type,
            distribution=args.distribution,
            truth_time_start=args.truth_time_start,
        )

    print(results)


if __name__ == "__main__":
    main()