#!/usr/bin/env python
"""
Plot train/valid loss for a fixed set of MultiTaskNet (task, h, noise) runs
on a single panel.

Skips configs that were only marked ``(selected)`` / not on disk yet.
Expects Hyak layout:
    <runs_dir>/<task_base>/multi_task_<task>_h*_n*_seed*/

Example:
    python scripts/mt_plot_selected_loss_curves.py \\
        /gscratch/golub/wong2/runs/multi_task \\
        --output /gscratch/golub/wong2/runs/multi_task/selected_loss_curves.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Selected accuracy-plot configs (exclude dms_anti / dmc_anti / dnms / dnmc).
SELECTED = [
    ("rt_go", 128, 0.05),
    ("rt_go_anti", 128, 0.0),
    ("dly_go", 256, 0.01),
    ("dly_go_anti", 256, 0.05),
    ("dms", 128, 0.0),
    ("dmc", 256, 0.001),
    ("ctxt_dm_1", 128, 0.001),
    ("ctxt_dm_2", 256, 0.0),
    ("ctxt_dm_max", 128, 0.05),
    ("dly_dm_1", 256, 0.1),
    ("dly_dm_2", 256, 0.01),
    ("dly_dm_max", 256, 0.01),
]


def find_event_file(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("events.out.tfevents*"))
    return files[0] if files else None


def parse_hn(name: str) -> tuple[int | None, float | None]:
    h_m = re.search(r"_h(\d+)(?=_|$)", name)
    n_m = re.search(r"_n(\d+(?:\.\d+)?)(?=_|$)", name)
    h = int(h_m.group(1)) if h_m else None
    n = float(n_m.group(1)) if n_m else None
    return h, n


def find_run(runs_dir: Path, task: str, hidden: int, noise: float) -> Path | None:
    task_dir = runs_dir / task
    if not task_dir.is_dir():
        return None
    matches = []
    for run_dir in sorted(task_dir.glob("multi_task_*")):
        if not run_dir.is_dir() or find_event_file(run_dir) is None:
            continue
        h, n = parse_hn(run_dir.name)
        if h == hidden and n is not None and abs(n - noise) < 1e-12:
            matches.append(run_dir)
    if not matches:
        return None
    # Prefer exact seed0 if multiple; else first.
    for m in matches:
        if "_seed0_" in m.name:
            return m
    return matches[0]


def load_scalars(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    scalars = ea.Scalars(tag)
    return [s.step for s in scalars], [float(s.value) for s in scalars]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs_dir",
        type=str,
        help="multi_task root containing <task_base>/multi_task_* run dirs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="PNG path (default: <runs_dir>/selected_loss_curves.png).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        help="Metric stem under train/ and valid/ (default: loss).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else runs_dir / "selected_loss_curves.png"
    )
    train_tag = f"train/{args.metric}"
    valid_tag = f"valid/{args.metric}"

    loaded = []
    for task, h, n in SELECTED:
        run_dir = find_run(runs_dir, task, h, n)
        if run_dir is None:
            print(f"  MISSING: {task} h={h} n={n:g}", flush=True)
            continue
        event = find_event_file(run_dir)
        train_x, train_y = load_scalars(event, train_tag)
        valid_x, valid_y = load_scalars(event, valid_tag)
        if not train_x and not valid_x:
            print(f"  skip (no {args.metric}): {run_dir.name}", flush=True)
            continue
        label = f"{task}  h={h}, n={n:g}"
        loaded.append(
            dict(
                task=task,
                hidden=h,
                noise=n,
                label=label,
                run=run_dir.name,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
            )
        )
        print(f"  loaded: {label}  <- {run_dir.name}", flush=True)

    if not loaded:
        raise SystemExit("No matching runs found.")

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(loaded))]

    fig, ax = plt.subplots(figsize=(10, 6))
    for r, color in zip(loaded, colors):
        if r["train_x"]:
            ax.plot(
                r["train_x"],
                r["train_y"],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
            )
        if r["valid_x"]:
            ax.plot(
                r["valid_x"],
                r["valid_y"],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
            )

    ax.set_xlabel("step")
    ax.set_ylabel(args.metric)
    ax.set_title(f"Selected runs — train/valid {args.metric}")
    ax.grid(alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.6, label="train"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="valid"),
    ]
    for r, color in zip(loaded, colors):
        legend_handles.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=2.0, label=r["label"])
        )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=8,
        title="style / run",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {len(loaded)} curves → {output}", flush=True)


if __name__ == "__main__":
    main()
