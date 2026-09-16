#!/usr/bin/env python
"""Diagnose why MultiTaskNet h=128 accuracy lags h=64 on L2 sweeps.

For each run, prints train/val gap, last logged step, task-loss vs L2-loss, and
whether the curve looks collapsed vs still-improving. Also writes acc/loss PNGs
and a config-diff of one h=64 vs one h=128 run.

Example (on Klone):
    python scripts/mt_diagnose_hidden_size.py \\
        /gscratch/golub/wong2/runs/multi_task/l2_weight_runs
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mt_collect_l2_accuracy import find_event_file, parse_run_name  # noqa: E402

COLLAPSE_ACC = 0.45
CONFIG_KEYS = [
    "hidden_size",
    "lr_init",
    "l2_scale",
    "l2_init",
    "l2_increase",
    "l2_comm_scale",
    "noise",
    "noise_init",
    "noise_increase",
    "num_channels",
    "num_areas",
    "num_angles",
    "rnn_type",
    "batch_size",
]


def load_scalars(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    scalars = ea.Scalars(tag)
    return [s.step for s in scalars], [float(s.value) for s in scalars]


def last(values: list[float]) -> float:
    return float(values[-1]) if values else float("nan")


def best(values: list[float], *, maximize: bool) -> float:
    if not values:
        return float("nan")
    return float(max(values) if maximize else min(values))


def load_resolved(run_dir: Path) -> dict:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("model", cfg)
    dm = cfg.get("datamodule", {})
    trainer = cfg.get("trainer", {})
    out = {}
    for key in CONFIG_KEYS:
        if key in model:
            out[key] = model[key]
        elif key in dm:
            out[key] = dm[key]
    out["max_epochs"] = trainer.get("max_epochs")
    out["min_epochs"] = trainer.get("min_epochs")
    return out


def collect_run(run_dir: Path) -> dict | None:
    event_file = find_event_file(run_dir)
    if event_file is None:
        return None
    meta = parse_run_name(run_dir.name)
    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    acc_tag = next((t for t in tags if t.startswith("valid/acc_")), None)
    task = acc_tag.split("valid/acc_", 1)[1] if acc_tag else meta["task"]

    tr_acc_s, tr_acc = load_scalars(event_file, f"train/acc_{task}")
    va_acc_s, va_acc = load_scalars(event_file, f"valid/acc_{task}")
    _, tr_loss = load_scalars(event_file, "train/loss")
    va_loss_s, va_loss = load_scalars(event_file, "valid/loss")
    _, va_resp = load_scalars(event_file, "valid/loss_resp")
    _, va_l2 = load_scalars(event_file, "valid/loss_l2")
    _, tr_resp = load_scalars(event_file, "train/loss_resp")

    last_step = va_acc_s[-1] if va_acc_s else (va_loss_s[-1] if va_loss_s else float("nan"))
    n_epochs = len(va_acc) if va_acc else len(va_loss)
    gap = last(tr_acc) - last(va_acc)
    status = "ok"
    if last(va_acc) < COLLAPSE_ACC and last(tr_acc) < COLLAPSE_ACC:
        status = "collapsed"
    elif abs(gap) < 0.03 and last(va_acc) < 0.7:
        status = "underfit"
    elif gap > 0.08:
        status = "overfit"

    return dict(
        run_name=run_dir.name,
        hidden_size=meta["hidden_size"],
        l2_scale=meta["l2_scale"],
        task=task,
        train_acc_final=last(tr_acc),
        valid_acc_final=last(va_acc),
        valid_acc_best=best(va_acc, maximize=True),
        acc_gap=gap,
        train_loss_final=last(tr_loss),
        valid_loss_final=last(va_loss),
        valid_loss_resp=last(va_resp),
        train_loss_resp=last(tr_resp),
        valid_loss_l2=last(va_l2),
        last_step=last_step,
        n_logged=n_epochs,
        status=status,
        va_acc_s=va_acc_s,
        va_acc=va_acc,
        tr_acc_s=tr_acc_s,
        tr_acc=tr_acc,
        va_loss_s=va_loss_s,
        va_loss=va_loss,
        resolved=load_resolved(run_dir),
    )


def print_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    show = [
        "hidden_size",
        "l2_scale",
        "task",
        "status",
        "train_acc_final",
        "valid_acc_final",
        "acc_gap",
        "valid_loss_resp",
        "valid_loss_l2",
        "last_step",
        "n_logged",
    ]
    print("\n=== Per-run diagnosis ===", flush=True)
    with pd.option_context(
        "display.max_columns", 20, "display.width", 180, "display.float_format", "{:.4g}".format
    ):
        print(df[show].sort_values(["hidden_size", "task", "l2_scale"]).to_string(index=False), flush=True)

    print("\n=== Status counts by hidden_size ===", flush=True)
    print(df.groupby(["hidden_size", "status"]).size().unstack(fill_value=0).to_string(), flush=True)

    print("\n=== Mean final train vs valid acc by hidden_size ===", flush=True)
    print(
        df.groupby("hidden_size")[["train_acc_final", "valid_acc_final", "acc_gap", "valid_loss_resp"]]
        .mean()
        .to_string(),
        flush=True,
    )


def plot_curves(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    for task, task_rows in by_task.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)
        for row in sorted(task_rows, key=lambda r: (r["hidden_size"], r["l2_scale"])):
            color = "C0" if row["hidden_size"] == 64 else "C1"
            ls = "-" if row["hidden_size"] == 64 else "--"
            label = f"h={row['hidden_size']} l2={row['l2_scale']:g}"
            if row["va_acc_s"]:
                axes[0].plot(row["va_acc_s"], row["va_acc"], color=color, ls=ls, lw=1.2, alpha=0.85, label=label)
            if row["va_loss_s"]:
                axes[1].plot(row["va_loss_s"], row["va_loss"], color=color, ls=ls, lw=1.2, alpha=0.85)
        axes[0].set_title(f"{task} valid acc")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("acc")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].set_title(f"{task} valid loss")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("loss")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"{task}_h64_vs_h128.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"  saved {path}", flush=True)


def print_config_diff(rows: list[dict]) -> None:
    h64 = next((r for r in rows if r["hidden_size"] == 64), None)
    h128 = next((r for r in rows if r["hidden_size"] == 128), None)
    if not h64 or not h128:
        print("\nNo h=64 and h=128 pair for config diff.", flush=True)
        return
    print(
        f"\n=== resolved_config diff ===\n  h64:  {h64['run_name']}\n  h128: {h128['run_name']}",
        flush=True,
    )
    keys = sorted(set(h64["resolved"]) | set(h128["resolved"]))
    for key in keys:
        a, b = h64["resolved"].get(key), h128["resolved"].get(key)
        mark = "  " if a == b else "* "
        print(f"  {mark}{key}: {a}  vs  {b}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=str)
    parser.add_argument("--pattern", type=str, default="multi_task_*")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="PNG directory (default: <runs_dir>/hsize_diagnosis).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else runs_dir / "hsize_diagnosis"
    run_dirs = sorted(p for p in runs_dir.glob(args.pattern) if p.is_dir() and find_event_file(p))
    if not run_dirs:
        raise SystemExit(f"No runs under {runs_dir}")

    rows = []
    for run_dir in run_dirs:
        row = collect_run(run_dir)
        if row:
            rows.append(row)
            print(
                f"{row['hidden_size']:>3}  l2={row['l2_scale']:<8g}  {row['task']:<12}  "
                f"{row['status']:<9}  train={row['train_acc_final']:.3f}  "
                f"val={row['valid_acc_final']:.3f}  gap={row['acc_gap']:+.3f}",
                flush=True,
            )

    print_table(rows)
    print_config_diff(rows)
    plot_curves(rows, out_dir)
    csv_path = out_dir / "hsize_diagnosis.csv"
    pd.DataFrame({k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in rows).to_csv(
        csv_path, index=False
    )
    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
