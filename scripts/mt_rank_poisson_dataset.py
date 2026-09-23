#!/usr/bin/env python
"""Rank DGN MultiTaskNet runs as MR-LFADS datasets.

For each run with ``data.h5``, this script:

  1. Reads valid task accuracy from TensorBoard (``valid/acc_<task>``).
  2. Poisson-samples each region's continuous hidden activity at **10 ms / 40 Hz**
     (``dt=0.01``, ``rate_max=40``), the same mapping used by
     ``src/evals/eval_utils.activity_to_lam`` and ``mt_compute_decodability.py``.
  3. Linear-decodes the signals that region is *supposed* to carry, on the
     relevant trial epoch, from both continuous activity and Poisson counts.

Task names are the Yang 20-rule set (``utils.cognitive_tasks.YANG20_TASKS``).
The folder parser matches longest-first so ``dms_nogo`` is not read as ``dms``.

Expected local signals follow ``model.stim_input_areas`` / ``sacc_output_areas``
(defaults from ``configs/multi_task``):

    A0  fix          (full)
    A1  stim1, stim2, stim_angle  (stim; plus delay when the trial has a gap)
    A1  amp1, amp2   (stim; DM / context-DM / delay-DM only)
    A2  task         (full; skipped if constant / single-task)
    A3  resp, sacc   (response)

Communication checks (signal should survive the graph, not just the input area):

    A3  stim_angle   (stim, delay when present, and response)

Delay pairs are omitted on ``rt_*`` and ``fd_*`` (response overlaps the
stimulus, so the delay mask is empty).

Runs are ranked by
    ``rank_score = valid_acc_best + mean Poisson R² over expected pairs``
among runs with ``valid_acc_best >= --min-acc``. Collapsed runs are listed but
not chosen as datasets.

By default every Yang-20 task is ranked, **h64 only**. Named groups:

    --tasks all          every Yang-20 run (default)
    --tasks selected     the 14 already in IRCB-26
    --tasks missing      the six without an IRCB-26 dataset
    --tasks unreleased   same six (alias)

Example (Klone, all tasks, h64 only):
    python scripts/mt_rank_poisson_dataset.py \\
        /gscratch/golub/wong2/runs/multi_task/l2_random_runs \\
        --tasks all --hidden-size 64
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
from utils.cognitive_tasks import YANG20_TASKS  # noqa: E402

POISSON_DT = 0.01
POISSON_RATE_MAX = 40.0
POISSON_COL = f"score_poisson_dt{POISSON_DT}_rate{int(POISSON_RATE_MAX)}"

# Longest first so dms_nogo / rt_go_anti / dly_dm_mod_1 win over shorter prefixes.
KNOWN_TASKS = tuple(sorted(YANG20_TASKS, key=len, reverse=True))
YANG20_INDEX = {name: i for i, name in enumerate(YANG20_TASKS)}

# Already have Poisson configs / selected h64 runs.
SELECTED_TASKS = (
    "fd_go",
    "fd_go_anti",
    "rt_go",
    "rt_go_anti",
    "dly_go",
    "dly_go_anti",
    "ctxt_dm_1",
    "ctxt_dm_2",
    "dly_dm_1",
    "dly_dm_2",
    "dms",
    "dms_nogo",
    "dmc",
    "dmc_nogo",
)
MISSING_TASKS = tuple(t for t in YANG20_TASKS if t not in SELECTED_TASKS)
# All Yang-20 tasks that are not yet in IRCB-26 / cognitive_task_suite.
UNRELEASED_TASKS = (
    "dm_1",
    "dm_2",
    "dly_dm_mod_1",
    "dly_dm_mod_2",
    "ctxt_dm_max",
    "dly_dm_max",
)
TASK_GROUPS = {
    "all": YANG20_TASKS,
    "selected": SELECTED_TASKS,
    "missing": MISSING_TASKS,
    "unreleased": UNRELEASED_TASKS,
}


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


def _task_from_config(cfg: dict) -> str | None:
    names = None
    dm = cfg.get("datamodule")
    model = cfg.get("model")
    if isinstance(dm, dict):
        names = dm.get("task_names")
    if not names and isinstance(model, dict):
        names = model.get("task_names")
    if not names:
        names = cfg.get("task_names")
    if isinstance(names, str):
        names = [names]
    if not names:
        return None
    task = str(names[0])
    return task if task in YANG20_INDEX else None


def load_area_map(run_dir: Path) -> tuple[list[str], str, str | None]:
    """Return stim_input_areas, sacc area, and optional task from resolved config."""
    path = run_dir / "resolved_config.yaml"
    fallback = run_dir / "hparams.yaml"
    stim = ["A0", "A1", "A1", "A2"]
    sacc = "A3"
    cfg_path = path if path.is_file() else fallback
    if not cfg_path.is_file():
        return stim, sacc, None
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("model", cfg)
    stim = list(model.get("stim_input_areas") or stim)
    sacc_areas = model.get("sacc_output_areas") or ["3"]
    raw = str(sacc_areas[0])
    sacc = raw if raw.startswith("A") else f"A{raw}"
    return stim, sacc, _task_from_config(cfg)


def _has_delay_epoch(task: str) -> bool:
    """rt_* and fd_* have no stim-off / response-off gap."""
    return not (task.startswith("rt_") or task.startswith("fd_"))


def _is_dm_task(task: str) -> bool:
    return (
        task.startswith("dm_")
        or task.startswith("ctxt_dm_")
        or task.startswith("dly_dm_")
    )


def expected_pairs(task: str, stim_areas: list[str], sacc_area: str) -> list[dict]:
    """Build expected (area, target, epoch, kind) rows for this task."""
    fix_area = stim_areas[0] if len(stim_areas) > 0 else "A0"
    stim_area = stim_areas[1] if len(stim_areas) > 1 else "A1"
    task_area = stim_areas[3] if len(stim_areas) > 3 else "A2"
    delay_ok = _has_delay_epoch(task)

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
    if _is_dm_task(task):
        pairs.extend(
            [
                dict(decode_from=stim_area, target="amp1", epoch="stim", kind="local"),
                dict(decode_from=stim_area, target="amp2", epoch="stim", kind="local"),
            ]
        )
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
    stim_areas, sacc_area, cfg_task = load_area_map(run_dir)
    if cfg_task:
        task = cfg_task
    if task == "unknown":
        print(f"    [warn] could not parse Yang-20 task from {run_dir.name}", flush=True)
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
    out["_task_ord"] = out["task"].map(lambda t: YANG20_INDEX.get(t, len(YANG20_INDEX)))
    out = out.sort_values(["_task_ord", "rank_score"], ascending=[True, False]).drop(
        columns="_task_ord"
    )
    return out.reset_index(drop=True)


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
    present = list(dict.fromkeys(rank_df["task"].tolist()))
    present.sort(key=lambda t: YANG20_INDEX.get(t, len(YANG20_INDEX)))
    for task in present:
        grp = rank_df[rank_df["task"] == task]
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


def resolve_tasks(raw: list[str]) -> list[str]:
    """Expand group aliases (``all`` / ``selected`` / ``missing`` / ``unreleased``) or keep names."""
    expanded: list[str] = []
    for item in raw:
        expanded.extend(TASK_GROUPS[item] if item in TASK_GROUPS else (item,))
    return list(dict.fromkeys(expanded))


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
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["all"],
        help=(
            "Tasks to rank, or a group: all (default), selected (14 released), "
            "missing / unreleased (six without IRCB-26 datasets). "
            f"Missing: {', '.join(MISSING_TASKS)}."
        ),
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        nargs="+",
        default=[64],
        help="Only rank these hidden sizes. Default: 64 (skips h256).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    h5_name = args.h5
    out_dir = runs_dir
    tasks = resolve_tasks(args.tasks)
    unknown = [t for t in tasks if t not in YANG20_INDEX]
    if unknown:
        raise SystemExit(f"Unknown --tasks: {unknown}")
    h_tag = "_h64" if args.hidden_size == [64] else ""
    if set(tasks) == set(SELECTED_TASKS):
        default_csv = f"mt_poisson_dataset_pairs_selected{h_tag}.csv"
    elif set(tasks) == set(MISSING_TASKS) or set(tasks) == set(UNRELEASED_TASKS):
        default_csv = f"mt_poisson_dataset_pairs_missing{h_tag}.csv"
    else:
        default_csv = f"mt_poisson_dataset_pairs{h_tag}.csv"
    pair_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else out_dir / default_csv
    )
    rank_path = pair_path.with_name(pair_path.stem.replace("_pairs", "") + "_ranking.csv")
    if rank_path == pair_path:
        rank_path = pair_path.with_name("mt_poisson_dataset_ranking.csv")

    run_dirs = discover_runs(runs_dir, args.pattern, h5_name)
    run_dirs = [d for d in run_dirs if parse_task(d.name) in set(tasks)]
    if args.hidden_size:
        keep_h = set(args.hidden_size)
        run_dirs = [d for d in run_dirs if parse_hidden(d.name) in keep_h]
    if not run_dirs:
        raise SystemExit(
            f"No runs with {h5_name} under {runs_dir} for tasks {tasks}"
            + (f" hidden_size={args.hidden_size}" if args.hidden_size else "")
        )

    print(f"Tasks: {', '.join(tasks)}", flush=True)
    if args.hidden_size:
        print(f"Hidden size filter: {args.hidden_size}", flush=True)
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
