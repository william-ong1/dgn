#!/usr/bin/env python
"""
Compute Poisson/continuous decodability for MultiTaskNet runs.

Supports single-task runs of:
    rt_go, rt_go_anti, dly_go, dly_go_anti

For each run × area (A0–A3) × target × epoch, evaluate held-out linear R²
on continuous hidden activity and on Poissonized spike counts.

Targets (from data.h5 truth-* plus stim_angle = stim1+stim2):
    fix, stim1, stim2, amp1, amp2, task, resp, sacc, stim_angle

Epochs (per-trial time masks; delay is empty on rt_* tasks):
    full      — all timesteps
    stim      — stimulus on
    delay     — after stim ends, before response (dly_* only)
    response  — response nonzero

This script only dumps metrics. Ranking / choosing the best (h, noise) model
is left to later analysis.

Writes a long-form CSV (rewritten after every run):
    run_name, task, decode_from, target, epoch, swept, hidden_size, noise, seed,
    score_continuous,
    score_poisson_dt0.01_rate20,
    score_poisson_dt0.01_rate40,
    score_poisson_dt0.05_rate40
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import utils.common_utils as alys  # noqa: E402

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NOISE = 0.01
DEFAULT_SEED = 0

POISSON_CONFIGS = [
    {"dt": 0.01, "rate_max": 20},
    {"dt": 0.01, "rate_max": 40},
    {"dt": 0.05, "rate_max": 40},
]

DEFAULT_AREAS = ["A0", "A1", "A2", "A3"]
SUPPORTED_TASKS = ("rt_go", "rt_go_anti", "dly_go", "dly_go_anti")

TARGETS = (
    "fix",
    "stim1",
    "stim2",
    "stim_angle",
    "amp1",
    "amp2",
    "task",
    "resp",
    "sacc",
)
EPOCHS = ("full", "stim", "delay", "response")

# Skip masked decodes with too few (batch, time) samples.
MIN_MASKED_SAMPLES = 256

# Stim channels include small input noise off-target; angles are O(1) when on.
# Threshold must sit above the noise floor so stim/delay epochs are meaningful.
STIM_ON_THRESHOLD = 0.1


def parse_run_name(name: str) -> dict:
    """Pull task / h / n / seed from names like multi_task_rt_go_h64_n0.01_seed0_id…"""
    task_m = re.search(r"multi_task_([a-z0-9_]+?)_h\d+", name)
    h_m = re.search(r"_h(\d+)(?=_|$)", name)
    n_m = re.search(r"_n(\d+(?:\.\d+)?)(?=_|$)", name)
    seed_m = re.search(r"_seed(\d+)(?=_|$)", name)

    hidden_size = int(h_m.group(1)) if h_m else DEFAULT_HIDDEN_SIZE
    noise = float(n_m.group(1)) if n_m else DEFAULT_NOISE
    seed = int(seed_m.group(1)) if seed_m else DEFAULT_SEED
    task = task_m.group(1) if task_m else "unknown"

    if n_m and h_m:
        swept = "hs_noise"
    elif n_m:
        swept = "noise"
    elif h_m:
        swept = "hidden_size"
    else:
        swept = "default"

    return dict(
        task=task,
        hidden_size=hidden_size,
        noise=noise,
        seed=seed,
        swept=swept,
    )


def sample_poisson_counts(activity, dt, rate_max, *, seed=0):
    """Map continuous hidden activity in ~[-1, 1] to Poisson spike counts."""
    rng = np.random.default_rng(seed)
    x = np.asarray(activity, dtype=np.float64)
    sd = float(np.std(x))
    if sd >= 1e-12:
        x = (x - float(np.mean(x))) / sd
    rates = rate_max * (x + 1.0) / 2.0
    lam = np.clip(rates * dt, a_min=0.0, a_max=rate_max)
    return rng.poisson(lam).astype(np.float32)


def _flatten_masked(X, y, mask):
    """Flatten (B,T,*) to (N,*), keeping only mask==True timesteps."""
    m = np.asarray(mask, dtype=bool)
    return X[m], y[m]


def decode_score(X, y, mask=None, *, degree=1, test_size=0.2, random_state=0):
    """Held-out R2 for linear (poly) decoding of y from X.

    X: (B, T, F), y: (B, T, C), mask: optional (B, T) bool.
    Splits by batch, then uses only masked timesteps within each split.
    Returns NaN if either split has too few masked samples.
    """
    idx = np.arange(X.shape[0])
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)

    if mask is None:
        mask = np.ones(X.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != X.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} != X batch/time {X.shape[:2]}")

    X_tr, y_tr = _flatten_masked(X[tr], y[tr], mask[tr])
    X_te, y_te = _flatten_masked(X[te], y[te], mask[te])
    if X_tr.shape[0] < MIN_MASKED_SAMPLES or X_te.shape[0] < max(32, int(0.05 * MIN_MASKED_SAMPLES)):
        return float("nan")

    reg = alys.PolyRegression(degree, alpha=1.0)
    # Fit on already-flat masked samples (bypass ffit's full flatten).
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    return float(r2_score(y_te, pred))


def _as_btc(arr, *, B: int, T: int) -> np.ndarray:
    """Normalize truth arrays to (B, T, C). Trial-level (B, C) is tiled over time."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(B, 1)
    if x.ndim == 2:
        if x.shape == (B, T):
            return x[:, :, None]
        if x.shape[0] == B:
            return np.broadcast_to(x[:, None, :], (B, T, x.shape[1])).copy()
    if x.ndim == 3 and x.shape[0] == B:
        if x.shape[1] == T:
            return x
        if x.shape[1] == 1:
            return np.broadcast_to(x, (B, T, x.shape[2])).copy()
    raise ValueError(f"Cannot reshape array with shape {x.shape} to (B={B}, T={T}, C)")


def build_targets(g) -> dict[str, np.ndarray]:
    """Build decode targets from multi-task truth-* datasets."""
    fix = np.asarray(g["truth-fix"], dtype=np.float32)
    B, T = int(fix.shape[0]), int(fix.shape[1])

    stim1 = _as_btc(g["truth-stim1"], B=B, T=T)
    stim2 = _as_btc(g["truth-stim2"], B=B, T=T)
    resp = _as_btc(g["truth-resp"], B=B, T=T)
    sacc = _as_btc(g["truth-sacc"], B=B, T=T)
    task = _as_btc(g["truth-task"], B=B, T=T)
    amp1 = _as_btc(g["truth-amp1"], B=B, T=T)
    amp2 = _as_btc(g["truth-amp2"], B=B, T=T)
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


def build_epoch_masks(
    targets: dict[str, np.ndarray],
    *,
    task: str = "",
    stim_threshold: float = STIM_ON_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Per-trial boolean masks (B, T) for full / stim / delay / response epochs.

    Delay = after stimulus ends and before response starts. On ``rt_*`` tasks
    response begins with the stimulus, so the delay mask is forced empty.
    """
    stim_mag = np.abs(targets["stim1"][..., 0]) + np.abs(targets["stim2"][..., 0])
    stim_on = stim_mag > stim_threshold
    resp_on = np.abs(targets["resp"][..., 0]) > stim_threshold

    if task.startswith("rt_"):
        delay = np.zeros_like(stim_on, dtype=bool)
    else:
        had_stim = np.cumsum(stim_on.astype(np.int32), axis=1) > 0
        delay = had_stim & (~stim_on) & (~resp_on)

    return {
        "full": np.ones_like(stim_on, dtype=bool),
        "stim": stim_on,
        "delay": delay,
        "response": resp_on,
    }


def process_run(
    run_dir: Path, *, areas: list[str], seed: int, h5_name: str = "data.h5"
) -> list[dict]:
    """Return one row per (area, target, epoch) for this run."""
    meta = parse_run_name(run_dir.name)
    if meta["task"] not in SUPPORTED_TASKS:
        print(
            f"    [warn] task={meta['task']!r} not in {SUPPORTED_TASKS}; "
            "still decoding with the same targets/epochs.",
            flush=True,
        )

    with h5py.File(run_dir / h5_name, "r") as h:
        g = h["0"]
        targets = build_targets(g)
        masks = build_epoch_masks(targets, task=meta["task"])

        hiddens = {}
        for area in areas:
            area_key = f"area-{area}"
            if area_key not in g:
                raise KeyError(f"{run_dir.name}: missing {area_key}")
            hiddens[area] = g[area_key][:]
            rep_attr = g[area_key].attrs.get("representation", "continuous")
            if rep_attr != "continuous":
                print(
                    f"    [warn] {area_key} is '{rep_attr}', not continuous; "
                    "Poisson resampling assumes continuous activity in ~[-1, 1].",
                    flush=True,
                )

    # Mask coverage (helps interpret NaNs on rt_* delay).
    for ename, mask in masks.items():
        n = int(mask.sum())
        print(f"    epoch={ename:<8} masked_samples={n}", flush=True)

    rows = []
    for area in areas:
        hidden = hiddens[area]

        for epoch in EPOCHS:
            mask = masks[epoch]
            row_key = [(t, epoch) for t in TARGETS]
            per: dict[tuple[str, str], dict] = {
                (tname, epoch): dict(
                    run_name=run_dir.name,
                    decode_from=area,
                    target=tname,
                    epoch=epoch,
                    **meta,
                )
                for tname in TARGETS
            }

            for tname in TARGETS:
                y = targets[tname]
                t0 = time.time()
                score = decode_score(hidden, y, mask=mask, random_state=seed)
                per[(tname, epoch)]["score_continuous"] = score
                print(
                    f"    [{area}|{epoch:<8}] continuous → {tname:<11}  "
                    f"R2={score:+.4f}   ({time.time() - t0:5.1f}s)",
                    flush=True,
                )

            for cfg in POISSON_CONFIGS:
                label = f"dt{cfg['dt']}_rate{cfg['rate_max']}"
                t0 = time.time()
                counts = sample_poisson_counts(
                    hidden, dt=cfg["dt"], rate_max=cfg["rate_max"], seed=seed
                )
                sample_dt = time.time() - t0
                for tname in TARGETS:
                    y = targets[tname]
                    t0 = time.time()
                    score = decode_score(counts, y, mask=mask, random_state=seed)
                    per[(tname, epoch)][f"score_poisson_{label}"] = score
                    print(
                        f"    [{area}|{epoch:<8}] poisson dt={cfg['dt']:>4} "
                        f"rate={cfg['rate_max']:>3} → {tname:<11}  "
                        f"R2={score:+.4f}   "
                        f"(sample {sample_dt:4.1f}s + decode {time.time() - t0:4.1f}s)",
                        flush=True,
                    )

            rows.extend(per[k] for k in row_key)

    return rows


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    print("\n=== Row counts ===", flush=True)
    print(
        df.groupby(["task", "epoch"], dropna=False).size().rename("n_rows").to_string(),
        flush=True,
    )

    score = "score_continuous"
    if score in df.columns:
        print(f"\n=== Area × target ({score}, epoch=full, mean over runs) ===", flush=True)
        sub = df[df["epoch"] == "full"]
        if not sub.empty:
            piv = sub.pivot_table(
                index="decode_from", columns="target", values=score, aggfunc="mean"
            )
            cols = [t for t in TARGETS if t in piv.columns]
            print(piv.reindex(columns=cols).to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)

        print(
            f"\n=== Area × target ({score}, epoch=delay, mean over runs; "
            "NaN expected for rt_*) ===",
            flush=True,
        )
        sub = df[df["epoch"] == "delay"]
        if not sub.empty:
            piv = sub.pivot_table(
                index="decode_from", columns="target", values=score, aggfunc="mean"
            )
            cols = [t for t in TARGETS if t in piv.columns]
            print(piv.reindex(columns=cols).to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)


def discover_run_dirs(runs_dir: Path, patterns: list[str], h5_name: str) -> list[Path]:
    """Match run subdirs, or treat runs_dir itself as a single run if it has the H5."""
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
    parser.add_argument(
        "runs_dir",
        type=str,
        help="Parent dir of runs, or a single run dir that contains the H5 file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Defaults to <runs_dir>/mt_decodability_results.csv.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        nargs="+",
        default=[
            "rt_go/multi_task_rt_go_h*",
            "rt_go_anti/multi_task_rt_go_anti_*",
            "dly_go/multi_task_dly_go_h*",
            "dly_go_anti/multi_task_dly_go_anti_*",
        ],
        help=(
            "Glob patterns relative to runs_dir (ignored if runs_dir is a single run). "
            "Defaults match Hyak layout multi_task/{task}/<run>/data.h5. "
            "rt_go_h* avoids matching rt_go_anti_* (same for dly)."
        ),
    )
    parser.add_argument(
        "--h5",
        type=str,
        default="data.h5",
        help="H5 filename inside each run dir (default: data.h5).",
    )
    parser.add_argument(
        "--area",
        type=str,
        nargs="+",
        default=DEFAULT_AREAS,
        help="Area(s) to decode FROM (default: A0 A1 A2 A3).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    h5_name = args.h5
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else runs_dir / "mt_decodability_results.csv"
    )

    run_dirs = discover_run_dirs(runs_dir, args.pattern, h5_name)

    print(f"Found {len(run_dirs)} runs in {runs_dir}", flush=True)
    print(f"H5 file: {h5_name}", flush=True)
    print(f"Supported tasks: {list(SUPPORTED_TASKS)}", flush=True)
    print(f"Areas: {args.area}", flush=True)
    print(f"Targets ({len(TARGETS)}): {list(TARGETS)}", flush=True)
    print(f"Epochs ({len(EPOCHS)}): {list(EPOCHS)}", flush=True)
    print(
        f"Cells/run: {len(args.area) * len(TARGETS) * len(EPOCHS)} "
        f"(× continuous + {len(POISSON_CONFIGS)} Poisson configs)",
        flush=True,
    )
    print(f"Output: {output_path}\n", flush=True)

    rows: list[dict] = []
    t_start = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        meta = parse_run_name(run_dir.name)
        print(
            f"[{i:>2}/{len(run_dirs)}] {run_dir.name}  "
            f"(task={meta['task']}, h={meta['hidden_size']}, "
            f"n={meta['noise']}, seed={meta['seed']})",
            flush=True,
        )
        try:
            rows.extend(
                process_run(run_dir, areas=args.area, seed=args.seed, h5_name=h5_name)
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
