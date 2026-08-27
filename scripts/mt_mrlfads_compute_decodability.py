#!/usr/bin/env python
"""
Task decodability for MR-LFADS multi-task runs, split by held-in vs held-out neurons.

For each MR-LFADS run with a forward-pass outputs H5:
  1. Load reconstructed activity from ``{outputs_dir}/{run_name}_outputs.h5``
  2. Load ground-truth task variables from the run's training ``data.h5``
  3. Align truth time to predictions (``ic_enc_seq_len`` crop, same as eval)
  4. Decode task targets from held-in neurons, held-out neurons, and (optionally) all

Uses the same targets / epochs / decoder as ``mt_compute_decodability.py``.

Example::

    python scripts/mt_mrlfads_compute_decodability.py \\
        --runs-dir mrlfads_runs \\
        --outputs-dir mrlfads_runs/mt_eval_outputs/dly_go \\
        --run-glob "dly_go_kl*" \\
        --recursive \\
        --output mrlfads_runs/mt_eval_outputs/dly_go/mt_mrlfads_decodability.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evals.eval_utils import (  # noqa: E402
    discover_mrlfads_runs,
    infer_submission_pred_time_len,
    load_mrlfads_ic_enc_seq_len,
    load_session_arrays,
    resolve_mrlfads_run_data_h5,
    slice_truth_for_time_alignment,
)


def _load_mt_decodability_module():
    path = PROJECT_ROOT / "scripts" / "mt_compute_decodability.py"
    spec = importlib.util.spec_from_file_location("mt_compute_decodability", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mtd = _load_mt_decodability_module()

TARGETS = mtd.TARGETS
EPOCHS = mtd.EPOCHS
SUPPORTED_TASKS = mtd.SUPPORTED_TASKS
DEFAULT_AREAS = mtd.DEFAULT_AREAS
decode_score = mtd.decode_score
build_epoch_masks = mtd.build_epoch_masks
_as_btc = mtd._as_btc

NEURON_SETS = ("held_in", "held_out", "all")


def parse_mrlfads_run_name(name: str) -> dict:
    """Parse task and KL from names like ``dly_go_kl0001_id2608172233``."""
    task = "unknown"
    for cand in SUPPORTED_TASKS:
        if name.startswith(cand):
            task = cand
            break

    kl = float("nan")
    m = re.search(r"_kl(\d+)", name)
    if m:
        raw = m.group(1)
        kl = int(raw) / 10 ** len(raw)

    return {"task": task, "kl": kl}


def build_targets_from_map(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Same targets as ``mt_compute_decodability.build_targets``, from in-memory arrays."""
    fix = np.asarray(arrays["truth-fix"], dtype=np.float32)
    B, T = int(fix.shape[0]), int(fix.shape[1])

    stim1 = _as_btc(arrays["truth-stim1"], B=B, T=T)
    stim2 = _as_btc(arrays["truth-stim2"], B=B, T=T)
    resp = _as_btc(arrays["truth-resp"], B=B, T=T)
    sacc = _as_btc(arrays["truth-sacc"], B=B, T=T)
    task = _as_btc(arrays["truth-task"], B=B, T=T)
    amp1 = _as_btc(arrays["truth-amp1"], B=B, T=T)
    amp2 = _as_btc(arrays["truth-amp2"], B=B, T=T)
    fix = _as_btc(fix, B=B, T=T)

    return {
        "fix": fix,
        "stim1": stim1,
        "stim2": stim2,
        "stim_angle": stim1 + stim2,
        "amp1": amp1,
        "amp2": amp2,
        "task": task,
        "resp": resp,
        "sacc": sacc,
    }


def _held_out_indices(submission: dict[str, np.ndarray], area: str) -> np.ndarray:
    key = f"meta-held-out-neuron-indices-area-{area}"
    if key not in submission:
        return np.array([], dtype=np.int64)
    return np.asarray(submission[key], dtype=np.int64).reshape(-1)


def _select_neurons(
    activity: np.ndarray,
    hn_idx: np.ndarray,
    neuron_set: str,
) -> np.ndarray | None:
    """Return activity with neuron dim subset for held_in / held_out / all."""
    n = activity.shape[-1]
    if neuron_set == "all":
        return activity

    ho = np.zeros(n, dtype=bool)
    if hn_idx.size:
        ho[hn_idx] = True

    if neuron_set == "held_out":
        if not ho.any():
            return None
        return activity[..., ho]
    if neuron_set == "held_in":
        if ho.all():
            return None
        return activity[..., ~ho]
    raise ValueError(f"Unknown neuron_set: {neuron_set!r}")


def _required_truth_keys() -> tuple[str, ...]:
    return (
        "truth-fix",
        "truth-stim1",
        "truth-stim2",
        "truth-resp",
        "truth-sacc",
        "truth-task",
        "truth-amp1",
        "truth-amp2",
    )


def process_mrlfads_run(
    run_dir: Path,
    *,
    outputs_h5: Path,
    areas: list[str],
    neuron_sets: list[str],
    seed: int,
    truth_h5: Path | None = None,
) -> list[dict]:
    """Return one row per (area, neuron_set, target, epoch) for this run."""
    if not outputs_h5.is_file():
        raise FileNotFoundError(f"Missing outputs H5: {outputs_h5}")

    meta = parse_mrlfads_run_name(run_dir.name)
    truth_path = truth_h5 or resolve_mrlfads_run_data_h5(run_dir)

    representations: dict[str, str] = {}
    with h5py.File(outputs_h5, "r") as h5:
        g = h5["0"]
        for area in areas:
            key = f"area-{area}"
            if key in g:
                representations[area] = g[key].attrs.get("representation", "unknown")

    submission = load_session_arrays(outputs_h5)
    truth_full = load_session_arrays(truth_path)

    pred_t = infer_submission_pred_time_len(submission)
    truth_start = load_mrlfads_ic_enc_seq_len(run_dir)
    truth = slice_truth_for_time_alignment(
        truth_full,
        truth_time_start=truth_start,
        pred_time_len=pred_t,
    )

    missing = [k for k in _required_truth_keys() if k not in truth]
    if missing:
        raise KeyError(f"{run_dir.name}: missing truth keys after align: {missing}")

    targets = build_targets_from_map(truth)
    masks = build_epoch_masks(targets, task=meta["task"])

    for ename, mask in masks.items():
        print(f"    epoch={ename:<8} masked_samples={int(mask.sum())}", flush=True)

    rows: list[dict] = []
    for area in areas:
        area_key = f"area-{area}"
        if area_key not in submission:
            raise KeyError(f"{run_dir.name}: missing {area_key} in {outputs_h5.name}")

        activity = np.asarray(submission[area_key], dtype=np.float32)
        rep = representations.get(area, "unknown")
        hn_idx = _held_out_indices(submission, area)
        print(
            f"    {area}: N={activity.shape[-1]}, held_out={hn_idx.size}, "
            f"held_in={activity.shape[-1] - hn_idx.size}",
            flush=True,
        )

        for neuron_set in neuron_sets:
            X = _select_neurons(activity, hn_idx, neuron_set)
            if X is None:
                print(f"    skip neuron_set={neuron_set} for {area} (empty subset)", flush=True)
                continue

            for epoch in EPOCHS:
                mask = masks[epoch]
                for tname in TARGETS:
                    y = targets[tname]
                    t0 = time.time()
                    score = decode_score(X, y, mask=mask, random_state=seed)
                    rows.append(
                        {
                            "run_name": run_dir.name,
                            "task": meta["task"],
                            "kl": meta["kl"],
                            "decode_from": area,
                            "neuron_set": neuron_set,
                            "n_neurons": int(X.shape[-1]),
                            "target": tname,
                            "epoch": epoch,
                            "score": score,
                            "outputs_h5": str(outputs_h5),
                            "truth_h5": str(truth_path),
                            "representation": rep,
                        }
                    )
                    print(
                        f"    [{area}|{neuron_set:<9}|{epoch:<8}] → {tname:<11}  "
                        f"R2={score:+.4f}   ({time.time() - t0:5.1f}s)",
                        flush=True,
                    )

    return rows


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    print("\n=== Row counts ===", flush=True)
    print(
        df.groupby(["task", "neuron_set", "epoch"], dropna=False)
        .size()
        .rename("n_rows")
        .to_string(),
        flush=True,
    )

    sub = df[df["epoch"] == "full"]
    if sub.empty:
        return

    print("\n=== held_in vs held_out (epoch=full, stim_angle, mean over runs) ===", flush=True)
    piv = sub[sub["target"] == "stim_angle"].pivot_table(
        index=["task", "decode_from"],
        columns="neuron_set",
        values="score",
        aggfunc="mean",
    )
    print(piv.to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Directory containing MR-LFADS run folders (with configs/ and checkpoints).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help=(
            "Directory with ``{run_name}_outputs.h5`` from mrlfads_forward_h5. "
            "Default: <runs-dir> (outputs live next to each run)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Defaults to <outputs-dir>/mt_mrlfads_decodability_results.csv.",
    )
    parser.add_argument(
        "--run-glob",
        type=str,
        default=None,
        help="Optional glob filter on run directory names (e.g. 'dly_go_kl*').",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search nested subdirectories for MR-LFADS runs.",
    )
    parser.add_argument(
        "--area",
        type=str,
        nargs="+",
        default=DEFAULT_AREAS,
        help="Area(s) to decode FROM (default: A0 A1 A2 A3).",
    )
    parser.add_argument(
        "--neuron-set",
        type=str,
        nargs="+",
        choices=NEURON_SETS,
        default=["held_in", "held_out"],
        help="Neuron subsets to decode from (default: held_in held_out).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    outputs_dir = (
        Path(args.outputs_dir).expanduser().resolve()
        if args.outputs_dir
        else runs_dir
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else outputs_dir / "mt_mrlfads_decodability_results.csv"
    )

    run_dirs = discover_mrlfads_runs(
        runs_dir, run_glob=args.run_glob, recursive=args.recursive
    )

    print(f"Found {len(run_dirs)} MR-LFADS runs under {runs_dir}", flush=True)
    print(f"Outputs dir: {outputs_dir}", flush=True)
    print(f"Neuron sets: {args.neuron_set}", flush=True)
    print(f"Areas: {args.area}", flush=True)
    print(f"Targets: {list(TARGETS)}", flush=True)
    print(f"Epochs: {list(EPOCHS)}", flush=True)
    print(f"Output: {output_path}\n", flush=True)

    rows: list[dict] = []
    t_start = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        outputs_h5 = outputs_dir / f"{run_dir.name}_outputs.h5"
        meta = parse_mrlfads_run_name(run_dir.name)
        print(
            f"[{i:>2}/{len(run_dirs)}] {run_dir.name}  "
            f"(task={meta['task']}, kl={meta['kl']})",
            flush=True,
        )
        if not outputs_h5.is_file():
            print(f"    skip: missing {outputs_h5}", flush=True)
            continue
        try:
            rows.extend(
                process_mrlfads_run(
                    run_dir,
                    outputs_h5=outputs_h5,
                    areas=args.area,
                    neuron_sets=args.neuron_set,
                    seed=args.seed,
                )
            )
            pd.DataFrame(rows).to_csv(output_path, index=False)
        except Exception as e:
            print(f"    ERROR: {e!r}", flush=True)

    elapsed = time.time() - t_start
    print(
        f"\nFinished {len(run_dirs)} runs ({len(rows)} rows) in {elapsed / 60:.1f} min",
        flush=True,
    )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print_summary(df)
    print(f"\nWrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
