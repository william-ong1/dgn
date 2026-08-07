#!/usr/bin/env python
"""
Plot train/valid loss curves from TensorBoard event files across all runs.

Default: one overview figure faceted by hidden size (rows) with train|valid
columns; curves colored by noise. Optionally also write per-run PNGs.

Example (on klone):

    python plot_loss_curves.py \\
        --runs-dir /path/to/rt_go \\
        --output-dir /path/to/rt_go/loss_curves
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_file(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("events.out.tfevents*"))
    if files:
        return files[0]
    nested = sorted(run_dir.glob("**/events.out.tfevents*"))
    return nested[0] if nested else None


def load_scalars(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return [], []
    scalars = ea.Scalars(tag)
    return [s.step for s in scalars], [s.value for s in scalars]


def list_scalar_tags(event_file: Path) -> list[str]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    return sorted(ea.Tags().get("scalars", []))


def parse_run_hparams(run_name: str) -> dict:
    m = re.search(r"h(\d+)_n([0-9.]+)_seed(\d+)", run_name)
    if not m:
        return {"hidden": None, "noise": None, "seed": None, "label": run_name}
    return {
        "hidden": int(m.group(1)),
        "noise": float(m.group(2)),
        "seed": int(m.group(3)),
        "label": f"h={m.group(1)}, n={m.group(2)}, seed={m.group(3)}",
    }


def load_all_runs(
    run_dirs: list[Path],
    train_tag: str,
    valid_tag: str,
) -> list[dict]:
    loaded = []
    for run_dir in run_dirs:
        event_file = find_event_file(run_dir)
        if event_file is None:
            print(f"  skip (no events): {run_dir.name}")
            continue
        train_x, train_y = load_scalars(event_file, train_tag)
        valid_x, valid_y = load_scalars(event_file, valid_tag)
        if not train_x and not valid_x:
            available = list_scalar_tags(event_file)
            print(f"  skip (no {train_tag}/{valid_tag}): {run_dir.name}")
            if available:
                preview = ", ".join(available[:20])
                suffix = " ..." if len(available) > 20 else ""
                print(f"    available tags: {preview}{suffix}")
            continue
        hp = parse_run_hparams(run_dir.name)
        loaded.append(
            {
                "name": run_dir.name,
                "event": event_file,
                "train_x": train_x,
                "train_y": train_y,
                "valid_x": valid_x,
                "valid_y": valid_y,
                **hp,
            }
        )
        print(f"  loaded: {run_dir.name} ({hp['label']})")
    return loaded


def noise_colormap(noises: list[float]):
    """Color map over noise; log scale if all values are > 0."""
    uniq = sorted({float(n) for n in noises if n is not None})
    if not uniq:
        return None, None, []
    if min(uniq) > 0:
        norm = LogNorm(vmin=min(uniq), vmax=max(uniq))
    else:
        # include 0: shift to linear
        norm = Normalize(vmin=min(uniq), vmax=max(uniq))
    cmap = plt.get_cmap("viridis")
    return cmap, norm, uniq


def plot_overview(
    runs: list[dict],
    metric: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Facet by hidden size; left=train, right=valid; color=noise."""
    with_h = [r for r in runs if r["hidden"] is not None]
    if not with_h:
        # fallback: single overlay of everything
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, kind in zip(axes, ("train", "valid")):
            for i, r in enumerate(runs):
                x, y = r[f"{kind}_x"], r[f"{kind}_y"]
                if not x:
                    continue
                ax.plot(x, y, linewidth=1.2, alpha=0.85, label=r["label"])
            ax.set_title(f"{kind}/{metric}")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(metric)
        axes[1].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"All runs — {metric}", fontsize=12)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        print(f"  saved overview: {output_path}")
        return

    hiddens = sorted({r["hidden"] for r in with_h})
    noises = [r["noise"] for r in with_h if r["noise"] is not None]
    cmap, norm, uniq_noise = noise_colormap(noises)

    n_rows = len(hiddens)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(12, max(3.2 * n_rows, 4)),
        sharex=True,
        squeeze=False,
    )

    for row, h in enumerate(hiddens):
        subset = sorted(
            [r for r in with_h if r["hidden"] == h],
            key=lambda r: (r["noise"] if r["noise"] is not None else -1, r["seed"] or 0),
        )
        for col, kind in enumerate(("train", "valid")):
            ax = axes[row][col]
            for r in subset:
                x, y = r[f"{kind}_x"], r[f"{kind}_y"]
                if not x:
                    continue
                color = cmap(norm(r["noise"])) if cmap is not None else f"C{subset.index(r) % 10}"
                ax.plot(x, y, color=color, linewidth=1.3, alpha=0.9)
            ax.set_title(f"h={h}  {kind}/{metric}" if row == 0 else f"h={h}  {kind}")
            ax.grid(alpha=0.3)
            if row == n_rows - 1:
                ax.set_xlabel("step")
            if col == 0:
                ax.set_ylabel(metric)

    if cmap is not None and uniq_noise:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("noise")
        # readable ticks at actual noise values
        cbar.set_ticks(uniq_noise)
        cbar.set_ticklabels([str(n) for n in uniq_noise])

    fig.suptitle(f"All runs — train vs valid {metric}", fontsize=13, y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved overview: {output_path}")


def plot_one_run(run: dict, metric: str, output_path: Path, dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if run["train_x"]:
        ax.plot(run["train_x"], run["train_y"], color="C0", linestyle="-", linewidth=1.5, label=f"train/{metric}")
    if run["valid_x"]:
        ax.plot(run["valid_x"], run["valid_y"], color="C1", linestyle="--", linewidth=1.5, label=f"valid/{metric}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{run['name']}\n{run['label']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing run folders (each with events.out.tfevents*).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs (default: <runs-dir>/loss_curves).",
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
        help="Glob for run subdirectories (default: multi_task_*).",
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit run folder names (relative to --runs-dir).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Also save one PNG per run (default: overview only).",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="Print scalar tags for the first matching run and exit.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"runs-dir not found: {runs_dir}")

    output_dir = (args.output_dir or (runs_dir / "loss_curves")).resolve()
    train_tag = f"train/{args.metric}"
    valid_tag = f"valid/{args.metric}"

    if args.run:
        run_dirs = [runs_dir / name for name in args.run]
    else:
        run_dirs = sorted(p for p in runs_dir.glob(args.pattern) if p.is_dir())

    if not run_dirs:
        raise SystemExit(f"No run dirs matching {args.pattern!r} under {runs_dir}")

    if args.list_tags:
        for run_dir in run_dirs:
            event_file = find_event_file(run_dir)
            if event_file is None:
                continue
            tags = list_scalar_tags(event_file)
            print(f"{run_dir.name} ({event_file.name})")
            for tag in tags:
                print(f"  {tag}")
            return
        raise SystemExit("No event files found to list tags from.")

    print(f"runs-dir:   {runs_dir}")
    print(f"output-dir: {output_dir}")
    print(f"tags:       {train_tag}, {valid_tag}")
    print(f"n_runs:     {len(run_dirs)}")

    runs = load_all_runs(run_dirs, train_tag, valid_tag)
    if not runs:
        raise SystemExit("No runs with usable loss scalars.")

    overview = output_dir / f"all_runs_{args.metric}.png"
    plot_overview(runs, args.metric, overview, dpi=args.dpi)

    if args.per_run:
        for run in runs:
            out = output_dir / f"{run['name']}_{args.metric}.png"
            plot_one_run(run, args.metric, out, dpi=args.dpi)
            print(f"  saved: {out}")

    print(f"done: overview of {len(runs)} runs → {overview}")


if __name__ == "__main__":
    main()
