#!/usr/bin/env python
"""
Plot MultiTaskNet train/valid loss curves — one figure per cognitive task.

Discovers Hyak-style layouts like ``mt_collect_accuracy.py``:
    multi_task/<task_base>/multi_task_*_h*_n*_seed*/events.out.tfevents*

For each task, writes one PNG (single axes). Each (hidden, noise) run gets a
distinct color; train=solid, valid=dashed.

Example:
    python scripts/mt_plot_loss_curves.py /gscratch/golub/wong2/runs/multi_task \\
        --output-dir /gscratch/golub/wong2/runs/multi_task/loss_curves

    python scripts/mt_plot_loss_curves.py /gscratch/golub/wong2/runs/multi_task/dly_go \\
        --output-dir /gscratch/golub/wong2/runs/multi_task/dly_go
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NOISE = 0.01
DEFAULT_SEED = 0


def parse_run_name(name: str) -> dict:
    task_m = re.search(r"multi_task_([a-z0-9_]+?)_h\d+", name)
    h_m = re.search(r"_h(\d+)(?=_|$)", name)
    n_m = re.search(r"_n(\d+(?:\.\d+)?)(?=_|$)", name)
    seed_m = re.search(r"_seed(\d+)(?=_|$)", name)
    return dict(
        task=task_m.group(1) if task_m else "unknown",
        hidden_size=int(h_m.group(1)) if h_m else DEFAULT_HIDDEN_SIZE,
        noise=float(n_m.group(1)) if n_m else DEFAULT_NOISE,
        seed=int(seed_m.group(1)) if seed_m else DEFAULT_SEED,
    )


def find_event_file(run_dir: Path) -> Path | None:
    """Events only in this directory (not recursive — safe for sweep roots)."""
    files = sorted(run_dir.glob("events.out.tfevents*"))
    return files[0] if files else None


def discover_run_dirs(runs_dir: Path, pattern: str) -> list[Path]:
    if find_event_file(runs_dir) is not None:
        return [runs_dir]

    direct = sorted(p for p in runs_dir.glob(pattern) if p.is_dir() and find_event_file(p))
    if direct:
        return direct

    nested: list[Path] = []
    for task_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(task_dir.glob(pattern)):
            if run_dir.is_dir() and find_event_file(run_dir):
                nested.append(run_dir)
    return nested


def task_base_for_run(run_dir: Path, runs_dir: Path) -> str:
    if run_dir.parent.resolve() == runs_dir.resolve():
        return parse_run_name(run_dir.name)["task"]
    return run_dir.parent.name


def load_scalars(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    scalars = ea.Scalars(tag)
    return [s.step for s in scalars], [float(s.value) for s in scalars]


def run_colors(runs: list[dict]) -> dict[tuple[int, float], tuple]:
    """Distinct color per (hidden_size, noise) pair."""
    keys = sorted({(r["hidden_size"], r["noise"]) for r in runs})
    n = len(keys)
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    return {k: cmap(i / max(n - 1, 1)) if n > 20 else cmap(i % cmap.N) for i, k in enumerate(keys)}


def plot_task_loss(
    task: str,
    runs: list[dict],
    metric: str,
    output_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """One PNG / one axes: color = (h, noise); train solid / valid dashed."""
    color_of = run_colors(runs)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ordered = sorted(runs, key=lambda r: (r["hidden_size"], r["noise"], r["seed"]))
    for r in ordered:
        color = color_of[(r["hidden_size"], r["noise"])]
        if r["train_x"]:
            ax.plot(
                r["train_x"],
                r["train_y"],
                color=color,
                linestyle="-",
                linewidth=1.3,
                alpha=0.9,
            )
        if r["valid_x"]:
            ax.plot(
                r["valid_x"],
                r["valid_y"],
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.9,
            )

    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title(f"{task} — train vs valid {metric}")
    ax.grid(alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.6, label="train"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="valid"),
    ]
    for (h, n), color in color_of.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=2.0,
                label=f"h={h}, n={n:g}",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        title="style / (h, n)",
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_path}  ({len(runs)} runs)", flush=True)


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
        "--output-dir",
        type=str,
        default=None,
        help="Where to write PNGs (default: <runs_dir>/loss_curves).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        help="Metric stem under train/ and valid/ (default: loss).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="multi_task_*",
        help="Glob for run subdirs (default: multi_task_*).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else runs_dir / "loss_curves"
    )
    train_tag = f"train/{args.metric}"
    valid_tag = f"valid/{args.metric}"

    run_dirs = discover_run_dirs(runs_dir, args.pattern)
    if not run_dirs:
        raise SystemExit(f"No run dirs with TensorBoard events under {runs_dir}")

    print(f"Found {len(run_dirs)} runs under {runs_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Tags:   {train_tag}, {valid_tag}\n", flush=True)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for i, run_dir in enumerate(run_dirs, 1):
        meta = parse_run_name(run_dir.name)
        task = task_base_for_run(run_dir, runs_dir)
        print(
            f"[{i:>3}/{len(run_dirs)}] {run_dir.name}  "
            f"(task={task}, h={meta['hidden_size']}, n={meta['noise']})",
            flush=True,
        )
        event_file = find_event_file(run_dir)
        if event_file is None:
            print("    skip: no events", flush=True)
            continue
        train_x, train_y = load_scalars(event_file, train_tag)
        valid_x, valid_y = load_scalars(event_file, valid_tag)
        if not train_x and not valid_x:
            print(f"    skip: no {train_tag}/{valid_tag}", flush=True)
            continue
        by_task[task].append(
            dict(
                name=run_dir.name,
                hidden_size=meta["hidden_size"],
                noise=meta["noise"],
                seed=meta["seed"],
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
            )
        )

    if not by_task:
        raise SystemExit("No runs with usable loss scalars.")

    print(f"\nPlotting {len(by_task)} task figure(s)...", flush=True)
    for task in sorted(by_task):
        out = output_dir / f"{task}_{args.metric}_curves.png"
        plot_task_loss(task, by_task[task], args.metric, out, dpi=args.dpi)

    print(f"\nDone: {len(by_task)} figures → {output_dir}", flush=True)


if __name__ == "__main__":
    main()
