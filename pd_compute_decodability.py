#!/usr/bin/env python
"""
Compute Poisson/continuous decodability for PassDecision runs.

For each run, decode from both ``area-P`` and ``area-D`` against three targets
derived from ``truth-inp`` (shape batch, time, 2):

    1. inp          — raw truth input
    2. cumsum       — cumulative sum of truth input along time
    3. sign_cumsum  — sign of that cumulative sum

Writes a long-form CSV (one row per run × area × target) with columns:
    run_name, decode_from, target, swept, hidden_size, noise_p, noise_d, ws,
    score_continuous,
    score_poisson_dt0.01_rate20,
    score_poisson_dt0.01_rate40,
    score_poisson_dt0.05_rate40

The CSV is rewritten after every run so partial results survive crashes.
"""
import argparse
import inspect
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# scripts/ -> repo root; expose src/ for utils.common_utils

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import utils.common_utils as alys  # noqa: E402

# --- Defaults from configs/pass_decision/model/model.yaml ---
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NOISE_P = 0.01
DEFAULT_NOISE_D = 0.01
DEFAULT_WS = 1.0

# --- Same Poisson configs as compute_decodability.py ---
POISSON_CONFIGS = [
    {"dt": 0.01, "rate_max": 20},
    {"dt": 0.01, "rate_max": 40},
    {"dt": 0.05, "rate_max": 40},
]

DEFAULT_AREAS = ["P", "D"]
TRUTH_KEY = "truth-inp"
TARGETS = ("inp", "cumsum", "sign_cumsum")


def parse_run_name(name: str) -> dict:
    """Pull h/n/ws out of names like pass_decision_h512_ws1.5_id2606091109.

    PassDecision uses ``noise_p`` / ``noise_d`` (not ``model.noise``). Run
    names from noise sweeps may encode ``_n{value}``; we store that in both
    noise_p and noise_d for metadata (the actual values live in resolved_config).
    """
    h_m = re.search(r"_h(\d+)(?=_|$)", name)
    n_m = re.search(r"_n(\d+(?:\.\d+)?)(?=_|$)", name)
    ws_m = re.search(r"_ws(\d+(?:\.\d+)?)(?=_|$)", name)

    hidden_size = int(h_m.group(1)) if h_m else DEFAULT_HIDDEN_SIZE
    noise = float(n_m.group(1)) if n_m else DEFAULT_NOISE_P
    ws = float(ws_m.group(1)) if ws_m else DEFAULT_WS

    if n_m:
        swept = "noise"
    elif ws_m:
        swept = "ws"
    else:
        swept = "default"

    return dict(
        hidden_size=hidden_size,
        noise_p=noise,
        noise_d=noise,
        ws=ws,
        swept=swept,
    )


def sample_poisson_counts(activity, dt, rate_max, *, seed=0):
    """Map continuous hidden activity in ~[-1, 1] to Poisson spike counts."""
    rng = np.random.default_rng(seed)
    rates = rate_max * (activity + 1.0) / 2.0
    lam = np.clip(rates * dt, a_min=0.0, a_max=rate_max)
    return rng.poisson(lam).astype(np.float32)


def decode_score(X, y, *, degree=1, test_size=0.2, random_state=0):
    """Held-out R2 for linear (poly) decoding of y from X.

    X: (B, T, F), y: (B, T, C) — C channels decoded jointly.
    """
    idx = np.arange(X.shape[0])
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
    reg = alys.PolyRegression(degree)
    reg.ffit(X[tr], y[tr])
    return _poly_r2_score(reg, X[te], y[te])


def _poly_r2_score(reg, X_test, y_test) -> float:
    """Held-out R2; works with old and new PolyRegression.fscore APIs."""
    if "metric" in inspect.signature(reg.fscore).parameters:
        return float(reg.fscore(X_test, y_test, metric="r2"))
    return float(reg.fscore(X_test, y_test))


def build_targets(inp: np.ndarray) -> dict[str, np.ndarray]:
    """Build decode targets from truth-inp (B, T, C)."""
    cumsum = np.cumsum(inp, axis=1)
    # Match PassDecision training: sign of cumulative input (keep zeros as 0)
    sign_cumsum = np.sign(cumsum).astype(np.float32)
    return {
        "inp": inp.astype(np.float32),
        "cumsum": cumsum.astype(np.float32),
        "sign_cumsum": sign_cumsum,
    }


def process_run(run_dir: Path, *, areas: list[str], seed: int) -> list[dict]:
    """Return one row per (area, target) for this run."""
    meta = parse_run_name(run_dir.name)

    with h5py.File(run_dir / "data.h5", "r") as h:
        g = h["0"]
        inp = g[TRUTH_KEY][:]  # (B, T, 2)
        hiddens = {}
        for area in areas:
            area_key = f"area-{area}"
            hiddens[area] = g[area_key][:]
            rep_attr = g[area_key].attrs.get("representation", "continuous")
            if rep_attr != "continuous":
                print(
                    f"    [warn] {area_key} is '{rep_attr}', not continuous; "
                    "Poisson resampling assumes continuous activity in ~[-1, 1].",
                    flush=True,
                )

    targets = build_targets(inp)
    rows = []

    for area in areas:
        hidden = hiddens[area]
        # One row per target, with continuous + poisson columns
        per_target: dict[str, dict] = {
            tname: dict(run_name=run_dir.name, decode_from=area, target=tname, **meta)
            for tname in TARGETS
        }

        for tname, y in targets.items():
            t0 = time.time()
            score = decode_score(hidden, y, random_state=seed)
            per_target[tname]["score_continuous"] = score
            print(
                f"    [{area}] continuous → {tname:<12}  R2={score:+.4f}"
                f"   ({time.time() - t0:5.1f}s)",
                flush=True,
            )

        for cfg in POISSON_CONFIGS:
            label = f"dt{cfg['dt']}_rate{cfg['rate_max']}"
            # Sample Poisson once per area/config, reuse across targets
            t0 = time.time()
            counts = sample_poisson_counts(
                hidden, dt=cfg["dt"], rate_max=cfg["rate_max"], seed=seed
            )
            sample_dt = time.time() - t0
            for tname, y in targets.items():
                t0 = time.time()
                score = decode_score(counts, y, random_state=seed)
                per_target[tname][f"score_poisson_{label}"] = score
                print(
                    f"    [{area}] poisson  dt={cfg['dt']:>4} rate={cfg['rate_max']:>3}"
                    f" → {tname:<12}  R2={score:+.4f}"
                    f"   (sample {sample_dt:4.1f}s + decode {time.time() - t0:4.1f}s)",
                    flush=True,
                )

        rows.extend(per_target[t] for t in TARGETS)

    return rows


def print_summary(df: pd.DataFrame) -> None:
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if df.empty or not score_cols:
        return

    print("\n=== ALL RUNS ===", flush=True)
    print(
        df.sort_values(
            ["swept", "hidden_size", "noise_p", "ws", "decode_from", "target"]
        ).to_string(index=False),
        flush=True,
    )

    for swept, sub in df.groupby("swept"):
        if swept == "default":
            continue
        var_col = {"noise": "noise_p", "ws": "ws"}[swept]
        print(f"\n=== Pivot by {swept} (rows=hidden_size, cols={var_col}) ===", flush=True)
        for area, area_sub in sub.groupby("decode_from"):
            for target, tgt_sub in area_sub.groupby("target"):
                for sc in score_cols:
                    piv = tgt_sub.pivot_table(index="hidden_size", columns=var_col, values=sc)
                    print(f"\n[{area} / {target} / {sc}]", flush=True)
                    print(piv.to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "runs_dir",
        type=str,
        help="Parent dir whose DIRECT children are runs (each containing data.h5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Defaults to <runs_dir>/pd_decodability_results.csv.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        nargs="+",
        default=["pass_decision_*"],
        help="One or more glob patterns for run subdirectories (deduplicated, non-recursive).",
    )
    parser.add_argument(
        "--area",
        type=str,
        nargs="+",
        default=DEFAULT_AREAS,
        help="Area(s) to decode FROM (default: P D).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else runs_dir / "pd_decodability_results.csv"
    )

    matched: dict[Path, None] = {}
    for pat in args.pattern:
        for d in runs_dir.glob(pat):
            if d.is_dir() and (d / "data.h5").is_file():
                matched[d] = None
    run_dirs = sorted(matched)

    print(f"Found {len(run_dirs)} runs in {runs_dir}", flush=True)
    print(
        f"Decoding targets {list(TARGETS)} from areas {args.area}",
        flush=True,
    )
    print(
        f"Defaults: hidden_size={DEFAULT_HIDDEN_SIZE}, noise_p/d={DEFAULT_NOISE_P}, "
        f"ws={DEFAULT_WS}",
        flush=True,
    )
    print(f"Output: {output_path}\n", flush=True)

    rows: list[dict] = []
    t_start = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        meta = parse_run_name(run_dir.name)
        print(
            f"[{i:>2}/{len(run_dirs)}] {run_dir.name}  "
            f"(swept={meta['swept']}, h={meta['hidden_size']}, "
            f"n={meta['noise_p']}, ws={meta['ws']})",
            flush=True,
        )
        try:
            rows.extend(process_run(run_dir, areas=args.area, seed=args.seed))
            pd.DataFrame(rows).to_csv(output_path, index=False)
        except Exception as e:
            print(f"    ERROR: {e!r}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nFinished {len(run_dirs)} runs ({len(rows)} rows) in {elapsed/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print_summary(df)
    print(f"\nWrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
