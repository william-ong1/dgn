#!/usr/bin/env python
"""
Compute decodability for MemoryNetwork / LIRC memory_network data.

For each run and area, evaluate held-out linear R² for:

    1. own_input   — area hidden → that area's private truth-inp channels
    2. msg_sent    — area hidden → inter-area messages this area sends
    3. msg_unsent  — area hidden → inter-area messages this area does *not* send
    4. msg_edge    — area hidden → each individual message channel (lag=0 detail)

Also lag-recovery scans (best lag compared to ground-truth ``lag`` from config):

    5. region_pair — for each inter-area edge src→dst, decode *source input*
       from *destination hidden* across lags. Expected peak at ``lag + 1``
       (message encodes inp[t-lag]; destination consumes it one step later).
    6. msg_vs_input — for each inter-area edge, decode source input from the
       message channel itself. Expected peak at ``lag`` (training-loss alignment).

Lag convention (matches training loss ``mesgs[:, lag:]`` vs ``inp[:, :-lag]``):
    lag > 0  → X[:, t] predicts y[:, t - lag]  (past target)
    lag = 0  → simultaneous

Message channel layout follows ``effectome = ranks * connectome`` (row = target,
column = source), flattened with zero-width slots dropped — same ordering as
``message-mesgs`` in the H5 export.

Writes a long-form CSV (rewritten after every run). Defaults load connectome /
ranks / lag from ``configs/memory_network``.
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import utils.common_utils as alys  # noqa: E402

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "memory_network"
TRUTH_KEY = "truth-inp"
MESG_KEY = "message-mesgs"


def load_mn_hparams(config_dir: Path) -> dict:
    model_cfg = config_dir / "model" / "model.yaml"
    with model_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "connectome": np.asarray(cfg["connectome"], dtype=np.int64),
        "ranks": np.asarray(cfg["ranks"], dtype=np.int64).reshape(-1),
        "lag": int(cfg.get("lag", 1)),
        "memory": int(cfg.get("memory", 1)),
        "hidden_size": int(cfg.get("hidden_size", -1)),
    }


def build_message_slots(connectome: np.ndarray, ranks: np.ndarray) -> list[dict]:
    """Map message-mesgs channel indices → (target, source) edges."""
    n = len(ranks)
    effectome = np.tile(ranks.reshape(1, -1), (n, 1)) * connectome
    mesgs_idx = effectome.flatten()
    slots: list[dict] = []
    offset = 0
    for ti in range(n):
        for sj in range(n):
            dim = int(mesgs_idx[ti * n + sj])
            if dim <= 0:
                continue
            slots.append(
                {
                    "ch": list(range(offset, offset + dim)),
                    "target": f"A{ti}",
                    "source": f"A{sj}",
                    "is_self": ti == sj,
                }
            )
            offset += dim
    return slots


def inter_area_edges(connectome: np.ndarray) -> list[tuple[str, str]]:
    """Directed non-self edges where connectome[dst, src] == 1."""
    n = connectome.shape[0]
    edges = []
    for dst in range(n):
        for src in range(n):
            if src != dst and int(connectome[dst, src]) != 0:
                edges.append((f"A{src}", f"A{dst}"))
    return edges


def area_input_slice(ranks: np.ndarray, area_idx: int) -> slice:
    start = int(ranks[:area_idx].sum())
    return slice(start, start + int(ranks[area_idx]))


def align_lag(X: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Align features/targets so X[t] predicts y[t - lag] (lag>0 = past target)."""
    if lag == 0:
        return X, y
    if lag > 0:
        return X[:, lag:], y[:, :-lag]
    L = -lag
    return X[:, :-L], y[:, L:]


def decode_score(X, y, *, degree=1, test_size=0.2, random_state=0) -> float:
    """Held-out R² for linear (poly) decoding of y from X. Shapes (B, T, F/C)."""
    if y.ndim == 2:
        y = y[..., None]
    if y.shape[-1] == 0 or X.shape[1] == 0:
        return float("nan")
    idx = np.arange(X.shape[0])
    tr, te = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True
    )
    reg = alys.PolyRegression(degree, alpha=1.0)
    reg.ffit(X[tr], y[tr])
    return _poly_r2_score(reg, X[te], y[te])


def _poly_r2_score(reg, X_test, y_test) -> float:
    if "metric" in inspect.signature(reg.fscore).parameters:
        return float(reg.fscore(X_test, y_test, metric="r2"))
    return float(reg.fscore(X_test, y_test))


def channels_from_slots(slots: list[dict]) -> list[int]:
    ch: list[int] = []
    for s in slots:
        ch.extend(s["ch"])
    return ch


def process_run(
    run_dir: Path,
    *,
    connectome: np.ndarray,
    ranks: np.ndarray,
    trained_lag: int,
    lags: list[int],
    seed: int,
    h5_name: str = "data.h5",
) -> list[dict]:
    slots = build_message_slots(connectome, ranks)
    area_names = [f"A{i}" for i in range(len(ranks))]
    edges = inter_area_edges(connectome)

    with h5py.File(run_dir / h5_name, "r") as h:
        g = h["0"]
        if TRUTH_KEY not in g:
            raise KeyError(f"Missing {TRUTH_KEY} in {run_dir / h5_name}")
        if MESG_KEY not in g:
            raise KeyError(f"Missing {MESG_KEY} in {run_dir / h5_name}")
        inp = g[TRUTH_KEY][:]  # (B, T, total_rank)
        mesgs = g[MESG_KEY][:]  # (B, T, total_mesgs)
        hiddens = {}
        for name in area_names:
            key = f"area-{name}"
            if key not in g:
                raise KeyError(f"Missing {key}")
            hiddens[name] = g[key][:]

    expected_mesg = sum(len(s["ch"]) for s in slots)
    if mesgs.shape[-1] != expected_mesg:
        print(
            f"    [warn] message-mesgs width {mesgs.shape[-1]} != "
            f"effectome total {expected_mesg}; check connectome/ranks config.",
            flush=True,
        )

    rows: list[dict] = []

    # --- Per-area: own input + sent vs unsent messages ---
    for ia, area in enumerate(area_names):
        hidden = hiddens[area]
        own_inp = inp[..., area_input_slice(ranks, ia)]

        # Inter-area only (exclude self/input slots from both sent and unsent)
        sent_slots = [s for s in slots if s["source"] == area and not s["is_self"]]
        unsent_slots = [s for s in slots if s["source"] != area and not s["is_self"]]
        sent_ch = channels_from_slots(sent_slots)
        unsent_ch = channels_from_slots(unsent_slots)

        targets: list[tuple[str, str, np.ndarray]] = [
            ("own_input", "own_input", own_inp),
            (
                "msg_sent",
                "msg_sent",
                mesgs[..., sent_ch] if sent_ch else np.zeros((*mesgs.shape[:2], 0)),
            ),
            (
                "msg_unsent",
                "msg_unsent",
                mesgs[..., unsent_ch] if unsent_ch else np.zeros((*mesgs.shape[:2], 0)),
            ),
        ]
        for s in slots:
            label = f"{s['source']}_to_{s['target']}"
            targets.append(("msg_edge", label, mesgs[..., s["ch"]]))

        for target_kind, target_name, y in targets:
            lag_list = lags if target_kind != "msg_edge" else [0]
            for lag in lag_list:
                t0 = time.time()
                X_a, y_a = align_lag(hidden, y, lag)
                score = decode_score(X_a, y_a, random_state=seed)
                n_ch = int(y.shape[-1]) if y.ndim == 3 else 1
                rows.append(
                    dict(
                        run_name=run_dir.name,
                        decode_from=area,
                        target_kind=target_kind,
                        target_name=target_name,
                        lag=lag,
                        n_channels=n_ch,
                        score=score,
                        is_trained_lag=int(lag == trained_lag),
                    )
                )
                tag = f"lag={lag:+d}" if target_kind != "msg_edge" else "lag=+0"
                print(
                    f"    [{area}] {target_kind:<11} {target_name:<16} "
                    f"{tag}  R2={score:+.4f}  ch={n_ch}  ({time.time() - t0:4.1f}s)",
                    flush=True,
                )

    # --- Lag recovery: dst hidden → src input, and message → src input ---
    print("    --- region-pair / msg-vs-input lag scan ---", flush=True)
    for src, dst in edges:
        src_idx = int(src[1:])
        src_inp = inp[..., area_input_slice(ranks, src_idx)]
        edge_slots = [
            s for s in slots if s["source"] == src and s["target"] == dst and not s["is_self"]
        ]
        msg_ch = channels_from_slots(edge_slots)
        msg = mesgs[..., msg_ch] if msg_ch else np.zeros((*mesgs.shape[:2], 0))

        for lag in lags:
            # Destination hidden → source's private input (cross-region info flow)
            t0 = time.time()
            X_a, y_a = align_lag(hiddens[dst], src_inp, lag)
            score = decode_score(X_a, y_a, random_state=seed)
            rows.append(
                dict(
                    run_name=run_dir.name,
                    decode_from=dst,
                    target_kind="region_pair",
                    target_name=f"{src}_input_in_{dst}",
                    lag=lag,
                    n_channels=int(src_inp.shape[-1]),
                    score=score,
                    is_trained_lag=int(lag == trained_lag),
                )
            )
            print(
                f"    [{dst}←{src}] region_pair  lag={lag:+d}  "
                f"R2={score:+.4f}  ({time.time() - t0:4.1f}s)",
                flush=True,
            )

            # Message channel → source input (training-loss alignment)
            t0 = time.time()
            X_a, y_a = align_lag(msg, src_inp, lag)
            score = decode_score(X_a, y_a, random_state=seed)
            rows.append(
                dict(
                    run_name=run_dir.name,
                    decode_from=f"msg:{src}_to_{dst}",
                    target_kind="msg_vs_input",
                    target_name=f"{src}_to_{dst}",
                    lag=lag,
                    n_channels=int(src_inp.shape[-1]),
                    score=score,
                    is_trained_lag=int(lag == trained_lag),
                )
            )
            print(
                f"    [msg {src}→{dst}] msg_vs_input lag={lag:+d}  "
                f"R2={score:+.4f}  ({time.time() - t0:4.1f}s)",
                flush=True,
            )

    return rows


def print_summary(df: pd.DataFrame, trained_lag: int) -> None:
    if df.empty:
        print("No results.", flush=True)
        return

    agg = df[df["target_kind"].isin(["own_input", "msg_sent", "msg_unsent"])]
    zero = agg[agg["lag"] == 0]
    print("\n=== Lag-0: own input / sent vs unsent ===", flush=True)
    if not zero.empty:
        piv = zero.pivot_table(
            index=["run_name", "decode_from"],
            columns="target_kind",
            values="score",
        )
        print(piv.to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)

    print("\n=== Best lag per (run, area, target) ===", flush=True)
    for (run, area, kind), sub in agg.groupby(
        ["run_name", "decode_from", "target_kind"]
    ):
        best = sub.loc[sub["score"].idxmax()]
        mark = " *" if int(best["lag"]) == trained_lag else ""
        print(
            f"  {run}/{area}/{kind}: best_lag={int(best['lag']):+d} "
            f"R2={best['score']:+.4f} (gt_lag={trained_lag:+d}){mark}",
            flush=True,
        )

    for kind, title, expected_lag in [
        (
            "region_pair",
            "Region-pair lag recovery (dst hidden → src input)",
            trained_lag + 1,  # message encodes inp[t-lag]; dst sees it one step later
        ),
        (
            "msg_vs_input",
            "Message→input lag recovery (mirrors training loss)",
            trained_lag,
        ),
    ]:
        sub_all = df[df["target_kind"] == kind]
        print(
            f"\n=== {title} (expected_lag={expected_lag:+d}, "
            f"config lag={trained_lag:+d}) ===",
            flush=True,
        )
        if sub_all.empty:
            continue
        for (run, edge), sub in sub_all.groupby(["run_name", "target_name"]):
            best = sub.loc[sub["score"].idxmax()]
            at_exp = sub.loc[sub["lag"] == expected_lag]
            exp_r2 = float(at_exp["score"].iloc[0]) if not at_exp.empty else float("nan")
            match = "MATCH" if int(best["lag"]) == expected_lag else "MISS"
            print(
                f"  {run}/{edge}: best_lag={int(best['lag']):+d} "
                f"R2={best['score']:+.4f} | expected={expected_lag:+d} "
                f"R2={exp_r2:+.4f}  [{match}]",
                flush=True,
            )

    print("\n=== Lag curves (own_input) ===", flush=True)
    own = agg[agg["target_kind"] == "own_input"]
    if not own.empty:
        piv = own.pivot_table(
            index="lag", columns=["run_name", "decode_from"], values="score"
        )
        print(piv.to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs_dir",
        type=str,
        help="Parent dir whose DIRECT children are runs (each containing the H5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Defaults to <runs_dir>/mn_decodability_results.csv.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        nargs="+",
        default=["gaussian", "poisson", "memory_network_*"],
        help="Glob patterns for run subdirectories (deduplicated, non-recursive).",
    )
    parser.add_argument(
        "--h5",
        type=str,
        default="data.h5",
        help="H5 filename inside each run dir (default: data.h5).",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="MemoryNetwork config dir with model/model.yaml (connectome, ranks, lag).",
    )
    parser.add_argument(
        "--lag-min",
        type=int,
        default=0,
        help="Inclusive min lag for sweeps (default: 0).",
    )
    parser.add_argument(
        "--lag-max",
        type=int,
        default=8,
        help="Inclusive max lag for sweeps (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()
    hps = load_mn_hparams(config_dir)
    lags = list(range(args.lag_min, args.lag_max + 1))
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else runs_dir / "mn_decodability_results.csv"
    )

    matched: dict[Path, None] = {}
    for pat in args.pattern:
        for d in runs_dir.glob(pat):
            if d.is_dir() and (d / args.h5).is_file():
                matched[d] = None
    run_dirs = sorted(matched)

    print(f"Found {len(run_dirs)} runs in {runs_dir}", flush=True)
    print(f"Config: {config_dir}", flush=True)
    print(
        f"  connectome=\n{hps['connectome']}\n"
        f"  ranks={hps['ranks'].tolist()}  trained_lag={hps['lag']}  "
        f"memory={hps['memory']}",
        flush=True,
    )
    print(f"Lag sweep: {lags}", flush=True)
    print(f"Output: {output_path}\n", flush=True)

    slots = build_message_slots(hps["connectome"], hps["ranks"])
    print("Message slots (message-mesgs channels):", flush=True)
    for s in slots:
        kind = "self/input" if s["is_self"] else "inter-area"
        print(
            f"  ch{s['ch']}: {s['source']} → {s['target']}  ({kind})",
            flush=True,
        )
    edges = inter_area_edges(hps["connectome"])
    print(
        "Inter-area edges for region-pair lag scan: "
        + ", ".join(f"{s}→{d}" for s, d in edges),
        flush=True,
    )
    print(flush=True)

    rows: list[dict] = []
    t_start = time.time()
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i:>2}/{len(run_dirs)}] {run_dir.name}", flush=True)
        try:
            rows.extend(
                process_run(
                    run_dir,
                    connectome=hps["connectome"],
                    ranks=hps["ranks"],
                    trained_lag=hps["lag"],
                    lags=lags,
                    seed=args.seed,
                    h5_name=args.h5,
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
    print_summary(df, trained_lag=hps["lag"])
    print(f"\nWrote: {output_path}", flush=True)


if __name__ == "__main__":
    main()
