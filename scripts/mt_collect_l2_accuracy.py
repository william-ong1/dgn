#!/usr/bin/env python
"""
Collect MultiTaskNet accuracy for l2_scale sweeps.

Reads TensorBoard tags ``train/acc_<task>`` and ``valid/acc_<task>`` from run
dirs named like:

    multi_task_rt_go_l21e-4_l2c0_gseed1_seed0_id2608272237

Prints one row per run, then a pivot of mean valid accuracy by l2_scale × gseed
so you can pick the best regularizer.

Example (on Klone):
    python scripts/mt_collect_l2_accuracy.py \\
        /gscratch/golub/wong2/runs/multi_task/l2_weight_runs
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# multi_task_<task>_l2<scale>_l2c<comm>_gseed<g>_seed<s>_id...
RUN_RE = re.compile(
    r"^multi_task_(?P<task>[a-z0-9_]+)"
    r"_l2(?P<l2>[0-9.eE+-]+)"
    r"_l2c(?P<l2c>[0-9.eE+-]+)"
    r"_gseed(?P<gseed>\d+)"
    r"_seed(?P<seed>\d+)"
)


def parse_run_name(name: str) -> dict:
    m = RUN_RE.search(name)
    if not m:
        return dict(task="unknown", l2_scale=float("nan"), l2_comm_scale=float("nan"),
                    gseed=float("nan"), seed=float("nan"))
    return dict(
        task=m.group("task"),
        l2_scale=float(m.group("l2")),
        l2_comm_scale=float(m.group("l2c")),
        gseed=int(m.group("gseed")),
        seed=int(m.group("seed")),
    )


def find_event_file(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("events.out.tfevents*"))
    return files[0] if files else None


def load_scalar_series(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    scalars = ea.Scalars(tag)
    return [s.step for s in scalars], [float(s.value) for s in scalars]


def summarize_series(steps: list[int], values: list[float], *, maximize: bool) -> dict:
    if not values:
        return dict(final=float("nan"), best=float("nan"), best_epoch=float("nan"))
    pick = max if maximize else min
    best_idx = pick(range(len(values)), key=lambda i: values[i])
    return dict(final=values[-1], best=values[best_idx], best_epoch=steps[best_idx])


def collect_run(run_dir: Path) -> list[dict]:
    event_file = find_event_file(run_dir)
    if event_file is None:
        print(f"    [warn] no TensorBoard events in {run_dir.name}", flush=True)
        return []

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    acc_tasks: dict[str, dict] = {}
    for tag in tags:
        m = re.fullmatch(r"(train|valid)/acc_(.+)", tag)
        if m:
            acc_tasks.setdefault(m.group(2), {})[m.group(1)] = tag

    if not acc_tasks:
        print(f"    [warn] no acc_* tags in {run_dir.name}", flush=True)
        return []

    meta = parse_run_name(run_dir.name)
    train_loss = summarize_series(*load_scalar_series(event_file, "train/loss"), maximize=False)
    valid_loss = summarize_series(*load_scalar_series(event_file, "valid/loss"), maximize=False)

    rows = []
    for task_name, split_tags in sorted(acc_tasks.items()):
        train = summarize_series(*load_scalar_series(event_file, split_tags.get("train", "")), maximize=True)
        valid = summarize_series(*load_scalar_series(event_file, split_tags.get("valid", "")), maximize=True)
        rows.append(
            dict(
                run_name=run_dir.name,
                task=task_name,
                l2_scale=meta["l2_scale"],
                l2_comm_scale=meta["l2_comm_scale"],
                gseed=meta["gseed"],
                seed=meta["seed"],
                train_acc_final=train["final"],
                train_acc_best=train["best"],
                train_acc_best_epoch=train["best_epoch"],
                valid_acc_final=valid["final"],
                valid_acc_best=valid["best"],
                valid_acc_best_epoch=valid["best_epoch"],
                train_loss_final=train_loss["final"],
                valid_loss_final=valid_loss["final"],
                valid_loss_best=valid_loss["best"],
                valid_loss_best_epoch=valid_loss["best_epoch"],
            )
        )
    return rows


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    n_l2 = int(df["l2_scale"].nunique(dropna=True))
    n_l2c = int(df["l2_comm_scale"].nunique(dropna=True))
    swept = "l2_comm_scale" if n_l2c > n_l2 else "l2_scale"

    show = [
        "l2_scale",
        "l2_comm_scale",
        "gseed",
        "task",
        "valid_acc_final",
        "valid_acc_best",
        "valid_acc_best_epoch",
        "valid_loss_final",
        "valid_loss_best",
        "train_acc_final",
    ]
    per_run = df.sort_values([swept, "gseed", "task"])
    print("\n=== Per-run accuracy ===", flush=True)
    with pd.option_context("display.max_columns", 20, "display.width", 160, "display.float_format", "{:.4g}".format):
        print(per_run[show].to_string(index=False), flush=True)

    print(f"\n=== Mean valid_acc_final by {swept} × gseed ===", flush=True)
    piv_final = df.pivot_table(index=swept, columns="gseed", values="valid_acc_final", aggfunc="mean")
    piv_final["mean"] = piv_final.mean(axis=1)
    with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
        print(piv_final.to_string(), flush=True)

    print(f"\n=== Mean valid_acc_best by {swept} × gseed ===", flush=True)
    piv_best = df.pivot_table(index=swept, columns="gseed", values="valid_acc_best", aggfunc="mean")
    piv_best["mean"] = piv_best.mean(axis=1)
    with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
        print(piv_best.to_string(), flush=True)

    by_swept = (
        df.groupby(swept, as_index=False)
        .agg(
            n_runs=("run_name", "nunique"),
            valid_acc_final_mean=("valid_acc_final", "mean"),
            valid_acc_best_mean=("valid_acc_best", "mean"),
            valid_loss_final_mean=("valid_loss_final", "mean"),
        )
        .sort_values("valid_acc_final_mean", ascending=False)
    )
    print("\n=== Ranked by mean valid_acc_final (across graph seeds) ===", flush=True)
    with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
        print(by_swept.to_string(index=False), flush=True)

    best_row = by_swept.iloc[0]
    print(
        f"\nBest {swept} by mean valid_acc_final: {best_row[swept]:.4g} "
        f"(mean={best_row['valid_acc_final_mean']:.4f}, n={int(best_row['n_runs'])})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs_dir",
        type=str,
        nargs="?",
        default="/gscratch/golub/wong2/runs/multi_task/l2_weight_runs",
        help="Directory of l2 sweep runs (default: Klone l2_weight_runs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path (default: <runs_dir>/mt_l2_accuracy_results.csv).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="multi_task_*",
        help="Glob for run subdirs (default: multi_task_*).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else runs_dir / "mt_l2_accuracy_results.csv"
    )

    run_dirs = sorted(
        p for p in runs_dir.glob(args.pattern) if p.is_dir() and find_event_file(p)
    )
    if not run_dirs:
        raise SystemExit(f"No run dirs with TensorBoard events under {runs_dir}")

    print(f"Found {len(run_dirs)} runs under {runs_dir}", flush=True)
    print(f"Output: {output_path}\n", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        meta = parse_run_name(run_dir.name)
        print(
            f"[{i:>3}/{len(run_dirs)}] {run_dir.name}  "
            f"(l2={meta['l2_scale']}, gseed={meta['gseed']})",
            flush=True,
        )
        try:
            run_rows = collect_run(run_dir)
            rows.extend(run_rows)
            if run_rows:
                r = run_rows[0]
                print(
                    f"    valid_acc final={r['valid_acc_final']:.4f} "
                    f"best={r['valid_acc_best']:.4f}",
                    flush=True,
                )
        except Exception as exc:
            print(f"    ERROR: {exc!r}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nFinished {len(run_dirs)} runs ({len(df)} rows) in {(time.time()-t0)/60:.1f} min", flush=True)
    print_summary(df)
    print(f"\nWrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
