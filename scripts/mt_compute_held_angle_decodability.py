#!/usr/bin/env python
"""
Compute held-angle decodability for MultiTaskNet runs.

Like ``mt_compute_decodability.py``, but the only decode target is ``held_angle``:
the trial's stimulus direction (from peak |stim1|+|stim2|), tiled over time.
This is the working-memory target for delay: unlike time-varying ``stim_angle``,
it stays nonzero after the stimulus turns off.

Supports single-task runs of:
    rt_go, rt_go_anti, dly_go, dly_go_anti

For each run × area (A0–A3) × epoch, evaluate held-out linear R² of
``held_angle`` from continuous hidden activity and Poissonized spike counts.

Epochs (same masks as ``mt_compute_decodability.py``; delay empty on rt_*):
    full, stim, delay, response

Writes a long-form CSV (rewritten after every run):
    run_name, task, decode_from, target, epoch, swept, hidden_size, noise, seed,
    score_continuous,
    score_poisson_dt0.01_rate20,
    score_poisson_dt0.01_rate40,
    score_poisson_dt0.05_rate40

Example:
    python scripts/mt_compute_held_angle_decodability.py /path/to/multi_task \\
        --pattern 'dly_go/multi_task_dly_go_h*' \\
        --output eval_outputs/dly_go/mt_held_angle_results.csv
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
TARGET = "held_angle"
EPOCHS = ("full", "stim", "delay", "response")

MIN_MASKED_SAMPLES = 256
STIM_ON_THRESHOLD = 0.1


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
    """Held-out R2 for linear (poly) decoding of y from X."""
    idx = np.arange(X.shape[0])
    tr, te = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True
    )

    if mask is None:
        mask = np.ones(X.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != X.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} != X batch/time {X.shape[:2]}")

    X_tr, y_tr = _flatten_masked(X[tr], y[tr], mask[tr])
    X_te, y_te = _flatten_masked(X[te], y[te], mask[te])
    if X_tr.shape[0] < MIN_MASKED_SAMPLES or X_te.shape[0] < max(
        32, int(0.05 * MIN_MASKED_SAMPLES)
    ):
        return float("nan")

    reg = alys.PolyRegression(degree, alpha=1.0)
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


def build_held_angle(stim1: np.ndarray, stim2: np.ndarray) -> np.ndarray:
    """Trial stimulus direction at peak |stim|, tiled over time → (B, T, 1)."""
    stim = stim1 + stim2
    B, T, _ = stim.shape
    mag = np.abs(stim1[..., 0]) + np.abs(stim2[..., 0])
    t_peak = np.argmax(mag, axis=1)
    held = stim[np.arange(B), t_peak, :]  # (B, C)
    return np.broadcast_to(held[:, None, :], (B, T, held.shape[1])).copy()


def build_epoch_masks(
    stim1: np.ndarray,
    stim2: np.ndarray,
    resp: np.ndarray,
    *,
    task: str = "",
    stim_threshold: float = STIM_ON_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Per-trial boolean masks (B, T) for full / stim / delay / response."""
    stim_mag = np.abs(stim1[..., 0]) + np.abs(stim2[..., 0])
    stim_on = stim_mag > stim_threshold
    resp_on = np.abs(resp[..., 0]) > stim_threshold

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
    """Return one row per (area, epoch) for held_angle on this run."""
    meta = parse_run_name(run_dir.name)
    if meta["task"] not in SUPPORTED_TASKS:
        print(
            f"    [warn] task={meta['task']!r} not in {SUPPORTED_TASKS}; "
            "still decoding held_angle with the same epochs.",
            flush=True,
        )

    with h5py.File(run_dir / h5_name, "r") as h:
        g = h["0"]
        fix = np.asarray(g["truth-fix"], dtype=np.float32)
        B, T = int(fix.shape[0]), int(fix.shape[1])
        stim1 = _as_btc(g["truth-stim1"], B=B, T=T)
        stim2 = _as_btc(g["truth-stim2"], B=B, T=T)
        resp = _as_btc(g["truth-resp"], B=B, T=T)
        held = build_held_angle(stim1, stim2)
        masks = build_epoch_masks(stim1, stim2, resp, task=meta["task"])

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

    for ename, mask in masks.items():
        print(f"    epoch={ename:<8} masked_samples={int(mask.sum())}", flush=True)

    # Sanity: held angle should vary across trials.
    held_trial = held[:, 0, 0]
    print(
        f"    held_angle trials: mean={held_trial.mean():+.3f} "
        f"std={held_trial.std():+.3f} "
        f"|peak_stim| mean={np.abs(held_trial).mean():+.3f}",
        flush=True,
    )

    rows = []
    for area in areas:
        hidden = hiddens[area]
        for epoch in EPOCHS:
            mask = masks[epoch]
            row = dict(
                run_name=run_dir.name,
                decode_from=area,
                target=TARGET,
                epoch=epoch,
                **meta,
            )

            t0 = time.time()
            score = decode_score(hidden, held, mask=mask, random_state=seed)
            row["score_continuous"] = score
            print(
                f"    [{area}|{epoch:<8}] continuous → {TARGET:<11}  "
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
                t0 = time.time()
                score = decode_score(counts, held, mask=mask, random_state=seed)
                row[f"score_poisson_{label}"] = score
                print(
                    f"    [{area}|{epoch:<8}] poisson dt={cfg['dt']:>4} "
                    f"rate={cfg['rate_max']:>3} → {TARGET:<11}  "
                    f"R2={score:+.4f}   "
                    f"(sample {sample_dt:4.1f}s + decode {time.time() - t0:4.1f}s)",
                    flush=True,
                )

            rows.append(row)

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
    print(
        f"\n=== Area × epoch ({score}, held_angle, mean over runs) ===",
        flush=True,
    )
    piv = df.pivot_table(
        index="decode_from", columns="epoch", values=score, aggfunc="mean"
    )
    cols = [e for e in EPOCHS if e in piv.columns]
    print(
        piv.reindex(columns=cols).to_string(float_format=lambda v: f"{v:+.4f}"),
        flush=True,
    )


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
        help="CSV output path. Defaults to <runs_dir>/mt_held_angle_results.csv.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        nargs="+",
        default=[
            "dly_go/multi_task_dly_go_h*",
            "dly_go_anti/multi_task_dly_go_anti_*",
            "rt_go/multi_task_rt_go_h*",
            "rt_go_anti/multi_task_rt_go_anti_*",
        ],
        help=(
            "Glob patterns relative to runs_dir (ignored if runs_dir is a single run). "
            "Defaults prefer dly_* (where delay memory matters)."
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
        else runs_dir / "mt_held_angle_results.csv"
    )

    run_dirs = discover_run_dirs(runs_dir, args.pattern, h5_name)

    print(f"Found {len(run_dirs)} runs in {runs_dir}", flush=True)
    print(f"H5 file: {h5_name}", flush=True)
    print(f"Supported tasks: {list(SUPPORTED_TASKS)}", flush=True)
    print(f"Areas: {args.area}", flush=True)
    print(f"Target: {TARGET} (trial stim angle tiled over time)", flush=True)
    print(f"Epochs ({len(EPOCHS)}): {list(EPOCHS)}", flush=True)
    print(
        f"Cells/run: {len(args.area) * len(EPOCHS)} "
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
