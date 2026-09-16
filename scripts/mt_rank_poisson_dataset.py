#!/usr/bin/env python
"""Rank DGN MultiTaskNet runs as MR-LFADS datasets.

For each run with ``data.h5``, this script:

  1. Reads valid task accuracy from TensorBoard (``valid/acc_<task>``).
  2. Poisson-samples each region's continuous hidden activity at **10 ms / 40 Hz**
     (``dt=0.01``, ``rate_max=40``), the same mapping used by
     ``src/evals/eval_utils.activity_to_lam`` and ``mt_compute_decodability.py``.
  3. Linear-decodes the signals that region is *supposed* to carry, on the
     relevant trial epoch, from both continuous activity and Poisson counts.

Expected local signals follow ``model.stim_input_areas`` / ``sacc_output_areas``
(defaults from ``configs/multi_task``):

    A0  fix          (full)
    A1  stim1, stim2, stim_angle  (stim; plus delay on dly_* tasks)
    A2  task         (full; skipped if constant / single-task)
    A3  resp, sacc   (response)

Communication checks (signal should survive the graph, not just the input area):

    A3  stim_angle   (stim, and delay on dly_* ; response on rt_*)

Runs are ranked by
    ``rank_score = valid_acc_best + mean Poisson R² over expected pairs``
among runs with ``valid_acc_best >= --min-acc``. Collapsed runs are listed but
not chosen as datasets.

Example (Klone):
    python scripts/mt_rank_poisson_dataset.py \\
        /gscratch/golub/wong2/runs/multi_task/l2_weight_runs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mt_collect_l2_accuracy import find_event_file, load_scalar_series, summarize_series  # noqa: E402
from mt_compute_decodability import (  # noqa: E402
    build_epoch_masks,
    build_targets,
    decode_score,
    sample_poisson_counts,
)

POISSON_DT = 0.01
POISSON_RATE_MAX = 40.0
POISSON_COL = f"score_poisson_dt{POISSON_DT}_rate{int(POISSON_RATE_MAX)}"

KNOWN_TASKS = (
    "rt_go_anti",
    "dly_go_anti",
    "rt_go",
    "dly_go",
    "dly_dm_max",
    "dly_dm_2",
    "dly_dm_1",
    "ctxt_dm_max",
    "ctxt_dm_2",
    "ctxt_dm_1",
    "dms",
)


def parse_task(name: str) -> str:
    rest = name
    if rest.startswith("multi_task_"):
        rest = rest[len("multi_task_") :]
    for task in KNOWN_TASKS:
        if rest == task or rest.startswith(task + "_"):
            return task
    return "unknown"


def parse_hidden(name: str) -> int:
    import re

    m = re.search(r"_h(\d+)(?=_|$)", name)
    return int(m.group(1)) if m else 64


def load_area_map(run_dir: Path) -> tuple[list[str], str]:
    """Return stim_input_areas and sacc area name from resolved config if present."""
    path = run_dir / "resolved_config.yaml"
    stim = ["A0", "A1", "A1", "A2"]
    sacc = "A3"
    if not path.is_file():
        return stim, sacc
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("model", cfg)
    stim = list(model.get("stim_input_areas") or stim)
    sacc_areas = model.get("sacc_output_areas") or ["3"]
    raw = str(sacc_areas[0])
    sacc = raw if raw.startswith("A") else f"A{raw}"
    return stim, sacc


def expected_pairs(task: str, stim_areas: list[str], sacc_area: str) -> list[dict]:
    """Build expected (area, target, epoch, kind) rows for this task."""
    fix_area = stim_areas[0] if len(stim_areas) > 0 else "A0"
    stim_area = stim_areas[1] if len(stim_areas) > 1 else "A1"
    task_area = stim_areas[3] if len(stim_areas) > 3 else "A2"
    delay_ok = not task.startswith("rt_")

    pairs = [
        dict(decode_from=fix_area, target="fix", epoch="full", kind="local"),
        dict(decode_from=stim_area, target="stim1", epoch="stim", kind="local"),
        dict(decode_from=stim_area, target="stim2", epoch="stim", kind="local"),
        dict(decode_from=stim_area, target="stim_angle", epoch="stim", kind="local"),
        dict(decode_from=task_area, target="task", epoch="full", kind="local"),
        dict(decode_from=sacc_area, target="resp", epoch="response", kind="local"),
        dict(decode_from=sacc_area, target="sacc", epoch="response", kind="local"),
        dict(decode_from=sacc_area, target="stim_angle", epoch="stim", kind="communication"),
        dict(decode_from=sacc_area, target="stim_angle", epoch="response", kind="communication"),
    ]
    if delay_ok:
        pairs.append(
            dict(decode_from=stim_area, target="stim_angle", epoch="delay", kind="local")
        )
        pairs.append(
            dict(
                decode_from=sacc_area,
                target="stim_angle",
                epoch="delay",
                kind="communication",
            )
        )
    # de-dup
    seen = set()
    out = []
    for p in pairs:
        key = (p["decode_from"], p["target"], p["epoch"])
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def load_valid_acc(run_dir: Path, task: str) -> dict:
    event_file = find_event_file(run_dir)
    if event_file is None:
        return dict(valid_acc_final=float("nan"), valid_acc_best=float("nan"))
    tag = f"valid/acc_{task}"
    summary = summarize_series(*load_scalar_series(event_file, tag), maximize=True)
    if not np.isfinite(summary["final"]):
        ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        ea.Reload()
        tags = [t for t in ea.Tags().get("scalars", []) if t.startswith("valid/acc_")]
        if tags:
            summary = summarize_series(*load_scalar_series(event_file, tags[0]), maximize=True)
    return dict(valid_acc_final=summary["final"], valid_acc_best=summary["best"])


def process_run(run_dir: Path, *, seed: int, h5_name: str) -> tuple[list[dict], dict]:
    task = parse_task(run_dir.name)
    hidden_size = parse_hidden(run_dir.name)
    stim_areas, sacc_area = load_area_map(run_dir)
    pairs = expected_pairs(task, stim_areas, sacc_area)
    acc = load_valid_acc(run_dir, task)

    with h5py.File(run_dir / h5_name, "r") as h:
        g = h["0"]
        targets = build_targets(g)
        masks = build_epoch_masks(targets, task=task)
        needed_areas = sorted({p["decode_from"] for p in pairs})
        hiddens = {}
        for area in needed_areas:
            key = f"area-{area}"
            if key not in g:
                raise KeyError(f"{run_dir.name}: missing {key}")
            hiddens[area] = np.asarray(g[key][:], dtype=np.float32)

    counts = {
        area: sample_poisson_counts(
            arr, dt=POISSON_DT, rate_max=POISSON_RATE_MAX, seed=seed
        )
        for area, arr in hiddens.items()
    }

    rows = []
    for spec in pairs:
        area, tname, epoch, kind = spec["decode_from"], spec["target"], spec["epoch"], spec["kind"]
        y = targets[tname]
        if float(np.var(y)) < 1e-12:
            cont = pois = float("nan")
        else:
            mask = masks[epoch]
            cont = decode_score(hiddens[area], y, mask=mask, random_state=seed)
            pois = decode_score(counts[area], y, mask=mask, random_state=seed)
        rows.append(
            dict(
                run_name=run_dir.name,
                task=task,
                hidden_size=hidden_size,
                decode_from=area,
                target=tname,
                epoch=epoch,
                kind=kind,
                score_continuous=cont,
                **{POISSON_COL: pois},
                poisson_keep=float(pois) / float(cont) if np.isfinite(cont) and abs(cont) > 1e-6 else float("nan"),
                **acc,
            )
        )
    return rows, acc


def rank_runs(pair_df: pd.DataFrame, *, min_acc: float) -> pd.DataFrame:
    def _mean(s):
        return float(np.nanmean(s.to_numpy(dtype=float))) if len(s) else float("nan")

    rows = []
    for run_name, g in pair_df.groupby("run_name"):
        local = g[g["kind"] == "local"]
        comm = g[g["kind"] == "communication"]
        acc_best = float(g["valid_acc_best"].iloc[0])
        acc_final = float(g["valid_acc_final"].iloc[0])
        pois_all = _mean(g[POISSON_COL])
        pois_local = _mean(local[POISSON_COL])
        pois_comm = _mean(comm[POISSON_COL])
        cont_all = _mean(g["score_continuous"])
        eligible = bool(np.isfinite(acc_best) and acc_best >= min_acc)
        rank_score = (acc_best + pois_all) if eligible and np.isfinite(pois_all) else float("-inf")
        rows.append(
            dict(
                run_name=run_name,
                task=g["task"].iloc[0],
                hidden_size=int(g["hidden_size"].iloc[0]),
                valid_acc_best=acc_best,
                valid_acc_final=acc_final,
                poisson_r2_mean=pois_all,
                poisson_r2_local=pois_local,
                poisson_r2_communication=pois_comm,
                continuous_r2_mean=cont_all,
                eligible=eligible,
                rank_score=rank_score,
            )
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["task", "rank_score"], ascending=[True, False])


def print_pair_table(df: pd.DataFrame) -> None:
    show = [
        "hidden_size",
        "task",
        "decode_from",
        "target",
        "epoch",
        "kind",
        "score_continuous",
        POISSON_COL,
        "valid_acc_best",
    ]
    print("\n=== Expected-signal decodability (continuous vs 10ms/40Hz Poisson) ===", flush=True)
    with pd.option_context("display.max_rows", 400, "display.width", 180, "display.float_format", "{:.4f}".format):
        print(df[show].to_string(index=False), flush=True)


def print_ranking(rank_df: pd.DataFrame, *, min_acc: float) -> None:
    print(
        f"\n=== Run ranking (eligible if valid_acc_best >= {min_acc:g}) ===",
        flush=True,
    )
    show = [
        "task",
        "hidden_size",
        "eligible",
        "valid_acc_best",
        "poisson_r2_mean",
        "poisson_r2_local",
        "poisson_r2_communication",
        "continuous_r2_mean",
        "rank_score",
        "run_name",
    ]
    with pd.option_context("display.max_colwidth", 80, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(rank_df[show].to_string(index=False), flush=True)

    print("\n=== Best dataset per task ===", flush=True)
    for task, grp in rank_df.groupby("task", sort=True):
        elig = grp[grp["eligible"]]
        pick = elig if not elig.empty else grp
        best = pick.iloc[0]
        print(
            f"  {task}: {best['run_name']}\n"
            f"      acc_best={best['valid_acc_best']:.4f}  "
            f"poisson_R2={best['poisson_r2_mean']:.4f}  "
            f"local={best['poisson_r2_local']:.4f}  "
            f"comm={best['poisson_r2_communication']:.4f}  "
            f"h={int(best['hidden_size'])}"
            + ("" if best["eligible"] else "  [below --min-acc; best of ineligible]"),
            flush=True,
        )


def discover_runs(runs_dir: Path, patterns: list[str], h5_name: str) -> list[Path]:
    if (runs_dir / h5_name).is_file():
        return [runs_dir]
    matched: dict[Path, None] = {}
    for pat in patterns:
        for d in runs_dir.glob(pat):
            if d.is_dir() and (d / h5_name).is_file():
                matched[d] = None
    return sorted(matched)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs_dir", type=str)
    parser.add_argument("--pattern", type=str, nargs="+", default=["multi_task_*"])
    parser.add_argument("--h5", type=str, default="data.h5")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-acc",
        type=float,
        default=0.6,
        help="Minimum valid_acc_best to be eligible as an MR-LFADS dataset.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    h5_name = args.h5
    out_dir = runs_dir
    pair_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else out_dir / "mt_poisson_dataset_pairs.csv"
    )
    rank_path = pair_path.with_name(pair_path.stem.replace("_pairs", "") + "_ranking.csv")
    if rank_path == pair_path:
        rank_path = pair_path.with_name("mt_poisson_dataset_ranking.csv")

    run_dirs = discover_runs(runs_dir, args.pattern, h5_name)
    if not run_dirs:
        raise SystemExit(f"No runs with {h5_name} under {runs_dir}")

    print(f"Found {len(run_dirs)} runs under {runs_dir}", flush=True)
    print(f"Poisson: dt={POISSON_DT}s ({POISSON_DT*1000:.0f} ms), rate_max={POISSON_RATE_MAX} Hz", flush=True)
    print(f"Pairs CSV: {pair_path}", flush=True)
    print(f"Rank  CSV: {rank_path}\n", flush=True)

    all_rows: list[dict] = []
    t0 = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i:>3}/{len(run_dirs)}] {run_dir.name}", flush=True)
        try:
            rows, acc = process_run(run_dir, seed=args.seed, h5_name=h5_name)
            all_rows.extend(rows)
            pois = np.nanmean([r[POISSON_COL] for r in rows])
            print(
                f"    acc_best={acc['valid_acc_best']:.4f}  "
                f"poisson_R2_mean={pois:.4f}  pairs={len(rows)}",
                flush=True,
            )
            pd.DataFrame(all_rows).to_csv(pair_path, index=False)
        except Exception as exc:
            print(f"    ERROR: {exc!r}", flush=True)

    pair_df = pd.DataFrame(all_rows)
    pair_df.to_csv(pair_path, index=False)
    if pair_df.empty:
        raise SystemExit("No successful runs.")

    rank_df = rank_runs(pair_df, min_acc=args.min_acc)
    rank_df.to_csv(rank_path, index=False)
    print_pair_table(pair_df)
    print_ranking(rank_df, min_acc=args.min_acc)
    print(f"\nFinished in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Wrote {pair_path}", flush=True)
    print(f"Wrote {rank_path}", flush=True)


if __name__ == "__main__":
    main()
