"""Run evaluation on a submission HDF5 file against the ground truth HDF5 file."""

import argparse
from pathlib import Path

import pandas as pd

from src.evals.evals import evaluate_submission
from src.evals.eval_utils import split_results_for_display


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
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=0,
        help="Number of bootstrap resamples for 95% CI (0 disables).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--rates-truth-h5",
        type=Path,
        default=None,
        help=(
            "Optional HDF5 with continuous area activity (e.g. gaussian.h5). "
            "When set, also computes rates-vs-rates R² (λ per bin) for Poisson runs."
        ),
    )
    parser.add_argument(
        "--poisson-dt",
        type=float,
        default=0.01,
        help="Bin width (s) used to convert continuous activity to λ = rate_hz * dt.",
    )
    parser.add_argument(
        "--poisson-rate-max",
        type=float,
        default=40.0,
        help="Max firing rate (Hz) used to convert continuous activity to λ.",
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
    if args.rates_truth_h5 is not None and not args.rates_truth_h5.expanduser().resolve().is_file():
        raise SystemExit(f"Rates truth HDF5 not found: {args.rates_truth_h5}")

    results = evaluate_submission(
        submission_h5=submission_h5,
        truth_h5=truth_h5,
        config_dir=config_dir,
        experiment_type=args.experiment_type,
        output_dist=args.output_dist,
        truth_time_start=args.truth_time_start,
        rates_truth_h5=args.rates_truth_h5,
        poisson_dt=args.poisson_dt,
        poisson_rate_max=args.poisson_rate_max,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )

    region_df, global_df, temporal_df = split_results_for_display(results)
    ci_map = results.get("confidence_intervals", {}) if isinstance(results, dict) else {}
    if isinstance(ci_map, dict) and ci_map:
        _attach_ci_columns(region_df, temporal_df, global_df, ci_map)

    if not region_df.empty:
        region_csv = region_df.copy()
        region_csv.insert(0, "section", "region")
    else:
        region_csv = pd.DataFrame(columns=["section"])

    if not global_df.empty:
        global_csv = global_df.copy()
        global_csv.insert(0, "section", "global")
    else:
        global_csv = pd.DataFrame(columns=["section"])

    if not temporal_df.empty:
        temporal_csv = temporal_df.copy()
        temporal_csv.insert(0, "section", "temporal")
    else:
        temporal_csv = pd.DataFrame(columns=["section"])

    dataframe = pd.concat([region_csv, temporal_csv, global_csv], ignore_index=True, sort=False)
    out_csv = args.output_csv.expanduser().resolve()
    dataframe.to_csv(out_csv, index=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print("Region-specific metrics:")
        if region_df.empty:
            print("(none)")
        else:
            print(region_df.to_string(index=False))
        print("\nTemporal metrics:")
        if temporal_df.empty:
            print("(none)")
        else:
            print(temporal_df.to_string(index=False))
        print("\nGlobal metrics:")
        if global_df.empty:
            print("(none)")
        else:
            print(global_df.to_string(index=False))
    print(f"Wrote to {out_csv}")

def _attach_ci_columns(
    region_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    global_df: pd.DataFrame,
    ci_map: dict,
) -> None:
    # Region metrics
    if not region_df.empty and "region" in region_df.columns:
        for col in [c for c in region_df.columns if c != "region"]:
            lows, highs = [], []
            for _, row in region_df.iterrows():
                region = row["region"]
                key = f"neural-activity.{region}.{col}"
                if col == "truth-inp-decode-r2":
                    key = f"truth-inp-decode.{region}"
                ci = ci_map.get(key, {})
                lows.append(ci.get("low", float("nan")))
                highs.append(ci.get("high", float("nan")))
            region_df[f"{col}_ci_low"] = lows
            region_df[f"{col}_ci_high"] = highs

    # Temporal metrics
    if not temporal_df.empty:
        row = temporal_df.iloc[0]
        for col in list(temporal_df.columns):
            ci = ci_map.get(f"temporal.{col}", {})
            temporal_df[f"{col}_ci_low"] = [ci.get("low", float("nan"))]
            temporal_df[f"{col}_ci_high"] = [ci.get("high", float("nan"))]

    # Global structure metrics and any top-level scalars
    if not global_df.empty:
        for col in list(global_df.columns):
            ci = ci_map.get(f"structure.{col}", None)
            if ci is None:
                ci = ci_map.get(f"aggregates.{col}", None)
            if ci is None:
                ci = ci_map.get(col, {})
            global_df[f"{col}_ci_low"] = [ci.get("low", float("nan"))]
            global_df[f"{col}_ci_high"] = [ci.get("high", float("nan"))]


if __name__ == "__main__":
    main()