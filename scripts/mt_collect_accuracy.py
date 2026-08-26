#!/usr/bin/env python
"""
Collect MultiTaskNet direction-classification accuracy from TensorBoard logs.

MultiTaskNet logs per-task accuracy at the last timestep as:
    train/acc_<task_name>
    valid/acc_<task_name>

Accuracy = fraction of trials where argmax(readout) == argmax(true response angle)
at the final time step, restricted to trials of that task.

Scans Hyak-style layouts, e.g.:
    multi_task/dly_go/multi_task_dly_go_h256_n0.05_seed0_id.../events.out.tfevents*

Writes a long-form CSV (one row per run × acc task × split summary):
    run_name, task_base, task, hidden_size, noise, seed,
    train_acc_final, train_acc_best, train_acc_best_epoch,
    valid_acc_final, valid_acc_best, valid_acc_best_epoch

Example:
    python scripts/mt_collect_accuracy.py /path/to/multi_task \\
        --output /path/to/multi_task/mt_accuracy_results.csv

    python scripts/mt_collect_accuracy.py /path/to/multi_task/dly_go \\
        --output /path/to/multi_task/dly_go/mt_accuracy_results.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NOISE = 0.01
DEFAULT_SEED = 0


def parse_run_name(name: str) -> dict:
    """Pull task / h / n / seed from names like multi_task_dly_go_h64_n0.01_seed0_id…"""
    task_m = re.search(r"multi_task_([a-z0-9_]+?)_h\d+", name)
    h_m = re.search(r"_h(\d+)(?=_|$)", name)
    n_m = re.search(r"_n(\d+(?:\.\d+)?)(?=_|$)", name)
    seed_m = re.search(r"_seed(\d+)(?=_|$)", name)

    hidden_size = int(h_m.group(1)) if h_m else DEFAULT_HIDDEN_SIZE
    noise = float(n_m.group(1)) if n_m else DEFAULT_NOISE
    seed = int(seed_m.group(1)) if seed_m else DEFAULT_SEED
    task = task_m.group(1) if task_m else "unknown"

    return dict(task=task, hidden_size=hidden_size, noise=noise, seed=seed)


def find_event_file(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("events.out.tfevents*"))
    if files:
        return files[0]
    nested = sorted(run_dir.glob("**/events.out.tfevents*"))
    return nested[0] if nested else None


def load_scalar_series(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    scalars = ea.Scalars(tag)
    return [s.step for s in scalars], [float(s.value) for s in scalars]


def summarize_series(steps: list[int], values: list[float]) -> dict:
    if not values:
        return dict(final=float("nan"), best=float("nan"), best_epoch=float("nan"))
    best_idx = max(range(len(values)), key=lambda i: values[i])
    return dict(
        final=values[-1],
        best=values[best_idx],
        best_epoch=steps[best_idx],
    )


def discover_run_dirs(runs_dir: Path, pattern: str) -> list[Path]:
    """Find run dirs under runs_dir or one level down (task_base/multi_task_*)."""
    if find_event_file(runs_dir) is not None:
        return [runs_dir]

    direct = sorted(p for p in runs_dir.glob(pattern) if p.is_dir() and find_event_file(p))
    if direct:
        return direct

    nested: dict[Path, None] = {}
    for task_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for run_dir in task_dir.glob(pattern):
            if run_dir.is_dir() and find_event_file(run_dir):
                nested[run_dir] = None
    return sorted(nested)


def task_base_for_run(run_dir: Path, runs_dir: Path) -> str:
    """Folder name for the task sweep, e.g. multi_task/dly_go/<run> -> dly_go."""
    if run_dir.parent.resolve() == runs_dir.resolve():
        return parse_run_name(run_dir.name)["task"]
    return run_dir.parent.name


def collect_run(run_dir: Path, *, runs_dir: Path) -> list[dict]:
    event_file = find_event_file(run_dir)
    if event_file is None:
        return []

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    acc_tasks: dict[str, dict] = {}
    for tag in tags:
        m = re.fullmatch(r"(train|valid)/acc_(.+)", tag)
        if not m:
            continue
        split, task_name = m.group(1), m.group(2)
        acc_tasks.setdefault(task_name, {})[split] = tag

    if not acc_tasks:
        print(f"    [warn] no acc_* tags in {run_dir.name}", flush=True)
        return []

    meta = parse_run_name(run_dir.name)
    base = task_base_for_run(run_dir, runs_dir)
    rows = []
    for task_name, split_tags in sorted(acc_tasks.items()):
        train = summarize_series(*load_scalar_series(event_file, split_tags.get("train", "")))
        valid = summarize_series(*load_scalar_series(event_file, split_tags.get("valid", "")))
        rows.append(
            dict(
                run_name=run_dir.name,
                task_base=base,
                task=task_name,
                hidden_size=meta["hidden_size"],
                noise=meta["noise"],
                seed=meta["seed"],
                train_acc_final=train["final"],
                train_acc_best=train["best"],
                train_acc_best_epoch=train["best_epoch"],
                valid_acc_final=valid["final"],
                valid_acc_best=valid["best"],
                valid_acc_best_epoch=valid["best_epoch"],
            )
        )
    return rows


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n=== Runs per task_base ===", flush=True)
    print(df.groupby("task_base").size().rename("n_rows").to_string(), flush=True)

    print("\n=== Mean valid_acc_best by task_base × h × n ===", flush=True)
    piv = df.pivot_table(
        index=["task_base", "hidden_size"],
        columns="noise",
        values="valid_acc_best",
        aggfunc="mean",
    )
    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print(piv.round(3).to_string(), flush=True)

    print("\n=== Top valid_acc_best per task_base ===", flush=True)
    idx = df.groupby("task_base")["valid_acc_best"].idxmax()
    cols = [
        "task_base",
        "run_name",
        "hidden_size",
        "noise",
        "valid_acc_best",
        "valid_acc_final",
    ]
    print(df.loc[idx, cols].sort_values("task_base").to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs_dir",
        type=str,
        help="multi_task root, a task folder (e.g. dly_go), or a single run dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path (default: <runs_dir>/mt_accuracy_results.csv).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="multi_task_*",
        help="Glob for run subdirs (default: multi_task_*).",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Print acc-related scalar tags for the first run and exit.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else runs_dir / "mt_accuracy_results.csv"
    )

    run_dirs = discover_run_dirs(runs_dir, args.pattern)
    if not run_dirs:
        raise SystemExit(f"No run dirs with TensorBoard events under {runs_dir}")

    if args.list_tags:
        event_file = find_event_file(run_dirs[0])
        ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        ea.Reload()
        print(f"{run_dirs[0].name}:")
        for tag in sorted(ea.Tags().get("scalars", [])):
            if "acc" in tag:
                print(f"  {tag}")
        return

    print(f"Found {len(run_dirs)} runs under {runs_dir}", flush=True)
    print(f"Output: {output_path}\n", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        meta = parse_run_name(run_dir.name)
        print(
            f"[{i:>3}/{len(run_dirs)}] {run_dir.name}  "
            f"(base={task_base_for_run(run_dir, runs_dir)}, "
            f"h={meta['hidden_size']}, n={meta['noise']}, seed={meta['seed']})",
            flush=True,
        )
        try:
            run_rows = collect_run(run_dir, runs_dir=runs_dir)
            rows.extend(run_rows)
            if run_rows:
                r = run_rows[0]
                print(
                    f"    valid_acc best={r['valid_acc_best']:.4f} "
                    f"final={r['valid_acc_final']:.4f}",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(output_path, index=False)
        except Exception as exc:
            print(f"    ERROR: {exc!r}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nFinished {len(run_dirs)} runs ({len(df)} rows) in {(time.time()-t0)/60:.1f} min", flush=True)
    print_summary(df)
    print(f"\nWrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
