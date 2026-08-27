#!/usr/bin/env python
"""
Cross-neuron prediction for MR-LFADS multi-task runs (held-in vs held-out).

MR-LFADS infers area **region factors** from encoder input (held-in neurons at
train time). The main readout maps those factors -> held-in activity; the
predictor maps factors -> held-out activity.

For each run with ``{run_name}_outputs.h5`` and aligned ground-truth ``data.h5``,
<<<<<<< Updated upstream
this script reports **both standard and McFadden R²** (same convention as
``batch_mrlfads_eval``: spikes in ``data.h5`` vs predicted rates):

**Factor -> activity (Ridge CV, trial split)**
  - ``factors_to_held_out`` / ``factors_to_held_in``

**Model heads** (from ``mrlfads_forward_h5`` exports when present)
  - ``readout_held_in`` / ``readout_held_out`` / ``predictor_held_out``

**Neural cross-predict (Ridge CV on ground-truth activity)**
  - ``held_in_to_held_out`` / ``held_out_to_held_in``

**Reconstruction** (merged submission, same as eval CSV)
  - ``recon_held_in`` / ``recon_held_out``

Each metric is written twice in the CSV with ``r2_type`` = ``standard`` or
``mcfadden``.
=======
this script reports:

**Factor -> activity (Ridge CV, trial split)**
  - ``factors_to_held_out`` — area factors predict held-out ground-truth activity
  - ``factors_to_held_in``  — area factors predict held-in ground-truth activity

**Model heads (when forward export includes readout / predictor tensors)**
  - ``readout_held_in`` / ``readout_held_out`` — main readout vs truth on each subset
  - ``predictor_held_out`` — predictor vs held-out truth (native held-out head)

**Neural cross-predict (Ridge CV on ground-truth activity)**
  - ``held_in_to_held_out`` — held-in activity predicts held-out activity
  - ``held_out_to_held_in`` — held-out activity predicts held-in activity

Note: a single forward pass yields one factor trajectory (inferred from the
held-in pathway). True ``factors inferred only from held-out neurons`` would
require re-running forward with a swapped ``hn_indices`` mask.
>>>>>>> Stashed changes

Example::

    python scripts/mt_mrlfads_compute_decodability.py \\
        --runs-dir mrlfads_runs \\
        --outputs-dir mrlfads_runs/mt_eval_outputs/dly_go \\
        --run-glob "dly_go_kl*" \\
        --recursive \\
        --output mrlfads_runs/mt_eval_outputs/dly_go/mt_mrlfads_cross_predict.csv
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
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

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
<<<<<<< Updated upstream
from src.evals.metrics import mcfadden_r2_poisson, standard_r2  # noqa: E402

SUPPORTED_TASKS = ("rt_go", "rt_go_anti", "dly_go", "dly_go_anti")
DEFAULT_AREAS = ("A0", "A1", "A2", "A3")
R2_TYPES = ("standard", "mcfadden")
MIN_RATE = 1e-8
=======
from src.evals.metrics import standard_r2  # noqa: E402

SUPPORTED_TASKS = ("rt_go", "rt_go_anti", "dly_go", "dly_go_anti")
DEFAULT_AREAS = ("A0", "A1", "A2", "A3")
>>>>>>> Stashed changes

METRICS = (
    "factors_to_held_out",
    "factors_to_held_in",
    "readout_held_out",
    "readout_held_in",
    "predictor_held_out",
    "held_in_to_held_out",
    "held_out_to_held_in",
    "recon_held_out",
    "recon_held_in",
)


def parse_mrlfads_run_name(name: str) -> dict:
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


def _held_out_indices(submission: dict[str, np.ndarray], area: str) -> np.ndarray:
    key = f"meta-held-out-neuron-indices-area-{area}"
    if key not in submission:
        return np.array([], dtype=np.int64)
    return np.asarray(submission[key], dtype=np.int64).reshape(-1)


def _held_out_mask(n_neurons: int, hn_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_neurons, dtype=bool)
    if hn_idx.size:
        mask[hn_idx] = True
    return mask


def _select_neurons(activity: np.ndarray, ho_mask: np.ndarray, subset: str) -> np.ndarray | None:
    if subset == "held_out":
        if not ho_mask.any():
            return None
        return activity[..., ho_mask]
    if subset == "held_in":
        if ho_mask.all():
            return None
        return activity[..., ~ho_mask]
    raise ValueError(f"Unknown subset: {subset!r}")


def _load_factor_slices(outputs_h5: Path, areas: list[str]) -> dict[str, slice]:
    """Return ``area -> slice`` into the ``region-factors`` last axis."""
    slices: dict[str, slice] = {}
    with h5py.File(outputs_h5, "r") as h5:
        g = h5["0"]
        offset = 0
        for area in areas:
            key = f"meta-factor-slice-area-{area}"
            if key in g:
                start, end = [int(v) for v in g[key][:]]
            else:
                if "region-factors" not in g:
                    raise KeyError(f"{outputs_h5.name}: missing region-factors")
                total = int(g["region-factors"].shape[-1])
                fac_dim = total // len(areas)
                if fac_dim * len(areas) != total:
                    raise ValueError(
                        f"{outputs_h5.name}: cannot infer equal factor slices "
                        f"(total={total}, n_areas={len(areas)})"
                    )
                start, end = offset, offset + fac_dim
                offset = end
            slices[area] = slice(start, end)
    return slices


def _load_area_fac_dims(run_dir: Path, areas: list[str]) -> dict[str, int] | None:
<<<<<<< Updated upstream
=======
    """Read per-area ``fac_dim`` from the saved MR-LFADS model config when present."""
>>>>>>> Stashed changes
    model_cfg = run_dir / "configs" / "model" / "model.yaml"
    if not model_cfg.is_file():
        return None

    with model_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    area_cfgs = cfg.get("areas") or cfg.get("area_configs")
    if not isinstance(area_cfgs, dict):
        return None

    out: dict[str, int] = {}
    for area in areas:
        ac = area_cfgs.get(area)
        if isinstance(ac, dict) and "fac_dim" in ac:
            out[area] = int(ac["fac_dim"])
    return out if len(out) == len(areas) else None


def _factor_slices_from_run(run_dir: Path, areas: list[str], factor_width: int) -> dict[str, slice]:
    fac_dims = _load_area_fac_dims(run_dir, areas)
    if fac_dims is not None and sum(fac_dims.values()) == factor_width:
        offset = 0
        slices: dict[str, slice] = {}
        for area in areas:
            dim = fac_dims[area]
            slices[area] = slice(offset, offset + dim)
            offset += dim
        return slices

    fac_dim = factor_width // len(areas)
    if fac_dim * len(areas) != factor_width:
        raise ValueError(
            f"region-factors width {factor_width} not divisible by n_areas={len(areas)}"
        )
    return {area: slice(i * fac_dim, (i + 1) * fac_dim) for i, area in enumerate(areas)}


<<<<<<< Updated upstream
def _rates_for_mcfadden(y_pred: np.ndarray) -> np.ndarray:
    """Clip predicted rates to a small positive floor for Poisson NLL."""
    return np.maximum(np.asarray(y_pred, dtype=np.float64), MIN_RATE)


def direct_r2_pair(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Standard + McFadden R² for direct array comparison.

    Matches ``neural_activity_reconstruction``: spike counts in ``y_true``,
    predicted rates in ``y_pred``.
    """
    rates = _rates_for_mcfadden(y_pred)
    return {
        "standard": float(standard_r2(y_true, y_pred)),
        "mcfadden": float(mcfadden_r2_poisson(y_true, rates)),
    }


def ridge_cross_r2_pair(
=======
def ridge_cross_r2(
>>>>>>> Stashed changes
    X: np.ndarray,
    Y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 0,
    alpha: float = 1.0,
<<<<<<< Updated upstream
) -> dict[str, float]:
    """Trial-held-out Ridge R² (standard + McFadden). Shapes (B, T, F) and (B, T, C)."""
    nan = {k: float("nan") for k in R2_TYPES}
    if Y.ndim == 2:
        Y = Y[..., None]
    if Y.shape[-1] == 0 or X.shape[-1] == 0 or X.shape[1] == 0:
        return nan
=======
) -> float:
    """Trial-held-out Ridge R² predicting Y from X. Shapes (B, T, F) and (B, T, C)."""
    if Y.ndim == 2:
        Y = Y[..., None]
    if Y.shape[-1] == 0 or X.shape[-1] == 0 or X.shape[1] == 0:
        return float("nan")
>>>>>>> Stashed changes

    idx = np.arange(X.shape[0])
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)

    x_tr = X[tr].reshape(-1, X.shape[-1])
    y_tr = Y[tr].reshape(-1, Y.shape[-1])
    x_te = X[te].reshape(-1, X.shape[-1])
<<<<<<< Updated upstream

    if x_tr.shape[0] < 32 or x_te.shape[0] < 16:
        return nan

    reg = Ridge(alpha=alpha)
    reg.fit(x_tr, y_tr)

    n_te = len(te)
    t_len = X.shape[1]
    n_ch = Y.shape[-1]
    y_te = Y[te].reshape(n_te, t_len, n_ch)
    pred = reg.predict(x_te).reshape(n_te, t_len, n_ch)
    rates = _rates_for_mcfadden(pred)

    return {
        "standard": float(standard_r2(y_te, pred)),
        "mcfadden": float(mcfadden_r2_poisson(y_te, rates)),
    }


def _append_score_rows(
    rows: list[dict],
    *,
    scores: dict[str, dict[str, float]],
    meta: dict,
    area: str,
    ho_mask: np.ndarray,
    fac_dim: int,
    outputs_h5: Path,
    truth_path: Path,
) -> None:
    for metric, by_type in scores.items():
        for r2_type in R2_TYPES:
            rows.append(
                {
                    "run_name": meta["run_name"],
                    "task": meta["task"],
                    "kl": meta["kl"],
                    "area": area,
                    "metric": metric,
                    "r2_type": r2_type,
                    "score": by_type.get(r2_type, float("nan")),
                    "n_held_in": int((~ho_mask).sum()),
                    "n_held_out": int(ho_mask.sum()),
                    "fac_dim": fac_dim,
                    "outputs_h5": str(outputs_h5),
                    "truth_h5": str(truth_path),
                }
            )
=======
    y_te = Y[te].reshape(-1, Y.shape[-1])

    if x_tr.shape[0] < 32 or x_te.shape[0] < 16:
        return float("nan")

    reg = Ridge(alpha=alpha)
    reg.fit(x_tr, y_tr)
    return float(standard_r2(y_te, reg.predict(x_te)))
>>>>>>> Stashed changes


def process_mrlfads_run(
    run_dir: Path,
    *,
    outputs_h5: Path,
    areas: list[str],
    seed: int,
    truth_h5: Path | None = None,
) -> list[dict]:
    if not outputs_h5.is_file():
        raise FileNotFoundError(f"Missing outputs H5: {outputs_h5}")

    meta = parse_mrlfads_run_name(run_dir.name)
    meta["run_name"] = run_dir.name
    truth_path = truth_h5 or resolve_mrlfads_run_data_h5(run_dir)

    submission = load_session_arrays(outputs_h5)
    truth_full = load_session_arrays(truth_path)

    if "region-factors" not in submission:
        raise KeyError(f"{run_dir.name}: missing region-factors in {outputs_h5.name}")

    pred_t = infer_submission_pred_time_len(submission)
    truth_start = load_mrlfads_ic_enc_seq_len(run_dir)
    truth = slice_truth_for_time_alignment(
        truth_full,
        truth_time_start=truth_start,
        pred_time_len=pred_t,
    )

    factors_all = np.asarray(submission["region-factors"], dtype=np.float32)
    try:
        factor_slices = _load_factor_slices(outputs_h5, areas)
    except (KeyError, ValueError):
        factor_slices = _factor_slices_from_run(run_dir, areas, factors_all.shape[-1])

    rows: list[dict] = []
    for area in areas:
        area_key = f"area-{area}"
        if area_key not in submission or area_key not in truth:
            print(f"    skip {area}: missing {area_key} in outputs or truth", flush=True)
            continue

        truth_act = np.asarray(truth[area_key], dtype=np.float32)
        recon_act = np.asarray(submission[area_key], dtype=np.float32)
        hn_idx = _held_out_indices(submission, area)
        ho_mask = _held_out_mask(truth_act.shape[-1], hn_idx)

        truth_hi = _select_neurons(truth_act, ho_mask, "held_in")
        truth_ho = _select_neurons(truth_act, ho_mask, "held_out")
        recon_hi = _select_neurons(recon_act, ho_mask, "held_in")
        recon_ho = _select_neurons(recon_act, ho_mask, "held_out")

        factors = factors_all[..., factor_slices[area]]

        print(
            f"    {area}: N={truth_act.shape[-1]}, held_out={int(ho_mask.sum())}, "
            f"held_in={int((~ho_mask).sum())}, fac_dim={factors.shape[-1]}",
            flush=True,
        )

<<<<<<< Updated upstream
        scores: dict[str, dict[str, float]] = {}

        if truth_ho is not None and factors.size:
            t0 = time.time()
            scores["factors_to_held_out"] = ridge_cross_r2_pair(
                factors, truth_ho, random_state=seed
            )
            s, m = scores["factors_to_held_out"]["standard"], scores["factors_to_held_out"]["mcfadden"]
            print(
                f"      factors -> held_out   std={s:+.4f}  mcfadden={m:+.4f}  "
=======
        scores: dict[str, float] = {}

        if truth_ho is not None and factors.size:
            t0 = time.time()
            scores["factors_to_held_out"] = ridge_cross_r2(
                factors, truth_ho, random_state=seed
            )
            print(
                f"      factors -> held_out   R2={scores['factors_to_held_out']:+.4f}  "
>>>>>>> Stashed changes
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

        if truth_hi is not None and factors.size:
            t0 = time.time()
<<<<<<< Updated upstream
            scores["factors_to_held_in"] = ridge_cross_r2_pair(
                factors, truth_hi, random_state=seed
            )
            s, m = scores["factors_to_held_in"]["standard"], scores["factors_to_held_in"]["mcfadden"]
            print(
                f"      factors -> held_in    std={s:+.4f}  mcfadden={m:+.4f}  "
=======
            scores["factors_to_held_in"] = ridge_cross_r2(
                factors, truth_hi, random_state=seed
            )
            print(
                f"      factors -> held_in    R2={scores['factors_to_held_in']:+.4f}  "
>>>>>>> Stashed changes
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

        readout_key = f"area-{area}-readout"
        pred_key = f"area-{area}-predictor-held-out"
        if readout_key in submission:
            readout = np.asarray(submission[readout_key], dtype=np.float32)
            if truth_hi is not None:
<<<<<<< Updated upstream
                readout_hi = _select_neurons(readout, ho_mask, "held_in")
                if readout_hi is not None:
                    scores["readout_held_in"] = direct_r2_pair(truth_hi, readout_hi)
            if truth_ho is not None:
                readout_ho = _select_neurons(readout, ho_mask, "held_out")
                if readout_ho is not None:
                    scores["readout_held_out"] = direct_r2_pair(truth_ho, readout_ho)
        else:
            print(f"      (no {readout_key}; re-run forward pass for readout metrics)", flush=True)

        if pred_key in submission and truth_ho is not None:
            predictor = np.asarray(submission[pred_key], dtype=np.float32)
            scores["predictor_held_out"] = direct_r2_pair(truth_ho, predictor)
        elif truth_ho is not None:
            print(f"      (no {pred_key}; re-run forward pass for predictor metrics)", flush=True)

        if truth_hi is not None and truth_ho is not None:
            t0 = time.time()
            scores["held_in_to_held_out"] = ridge_cross_r2_pair(
                truth_hi, truth_ho, random_state=seed
            )
            scores["held_out_to_held_in"] = ridge_cross_r2_pair(
                truth_ho, truth_hi, random_state=seed
            )
            hi_ho = scores["held_in_to_held_out"]
            ho_hi = scores["held_out_to_held_in"]
            print(
                f"      held_in -> held_out    std={hi_ho['standard']:+.4f}  "
                f"mcfadden={hi_ho['mcfadden']:+.4f}  "
                f"held_out -> held_in std={ho_hi['standard']:+.4f}  "
                f"mcfadden={ho_hi['mcfadden']:+.4f}  "
=======
                scores["readout_held_in"] = float(
                    standard_r2(truth_hi, _select_neurons(readout, ho_mask, "held_in"))
                )
            if truth_ho is not None:
                scores["readout_held_out"] = float(
                    standard_r2(truth_ho, _select_neurons(readout, ho_mask, "held_out"))
                )

        if pred_key in submission and truth_ho is not None:
            predictor = np.asarray(submission[pred_key], dtype=np.float32)
            scores["predictor_held_out"] = float(standard_r2(truth_ho, predictor))

        if truth_hi is not None and truth_ho is not None:
            t0 = time.time()
            scores["held_in_to_held_out"] = ridge_cross_r2(
                truth_hi, truth_ho, random_state=seed
            )
            scores["held_out_to_held_in"] = ridge_cross_r2(
                truth_ho, truth_hi, random_state=seed
            )
            print(
                f"      held_in -> held_out    R2={scores['held_in_to_held_out']:+.4f}  "
                f"held_out -> held_in R2={scores['held_out_to_held_in']:+.4f}  "
>>>>>>> Stashed changes
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

<<<<<<< Updated upstream
        if recon_hi is not None and truth_hi is not None:
            scores["recon_held_in"] = direct_r2_pair(truth_hi, recon_hi)
        if recon_ho is not None and truth_ho is not None:
            scores["recon_held_out"] = direct_r2_pair(truth_ho, recon_ho)

        _append_score_rows(
            rows,
            scores=scores,
            meta=meta,
            area=area,
            ho_mask=ho_mask,
            fac_dim=int(factors.shape[-1]),
            outputs_h5=outputs_h5,
            truth_path=truth_path,
        )
=======
        if recon_hi is not None:
            scores["recon_held_in"] = float(standard_r2(truth_hi, recon_hi))
        if recon_ho is not None:
            scores["recon_held_out"] = float(standard_r2(truth_ho, recon_ho))

        for metric, score in scores.items():
            rows.append(
                {
                    "run_name": run_dir.name,
                    "task": meta["task"],
                    "kl": meta["kl"],
                    "area": area,
                    "metric": metric,
                    "score": score,
                    "n_held_in": int((~ho_mask).sum()),
                    "n_held_out": int(ho_mask.sum()),
                    "fac_dim": int(factors.shape[-1]),
                    "outputs_h5": str(outputs_h5),
                    "truth_h5": str(truth_path),
                }
            )
>>>>>>> Stashed changes

    return rows


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

<<<<<<< Updated upstream
    print("\n=== Mean scores by task x metric x r2_type ===", flush=True)
    piv = df.pivot_table(
        index=["task", "metric"], columns="r2_type", values="score", aggfunc="mean"
=======
    print("\n=== Mean scores by task x metric ===", flush=True)
    piv = df.pivot_table(index=["task", "metric"], values="score", aggfunc="mean")
    print(piv.to_string(float_format=lambda v: f"{v:+.4f}"), flush=True)

    core = df[df["metric"].isin(["factors_to_held_out", "factors_to_held_in"])]
    if core.empty:
        return

    print("\n=== factors -> neurons (mean over runs/areas) ===", flush=True)
    print(
        core.pivot_table(index="task", columns="metric", values="score", aggfunc="mean")
        .to_string(float_format=lambda v: f"{v:+.4f}"),
        flush=True,
>>>>>>> Stashed changes
    )

    core = df[df["metric"].isin(["factors_to_held_out", "factors_to_held_in"])]
    if core.empty:
        return

    print("\n=== factors -> neurons (mean over runs/areas) ===", flush=True)
    print(
        core.pivot_table(
            index=["task", "metric"], columns="r2_type", values="score", aggfunc="mean"
        ).to_string(float_format=lambda v: f"{v:+.4f}"),
        flush=True,
    )


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
            "Default: <runs-dir>."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path. Defaults to <outputs-dir>/mt_mrlfads_cross_predict.csv.",
    )
    parser.add_argument("--run-glob", type=str, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--area",
        type=str,
        nargs="+",
        default=list(DEFAULT_AREAS),
        help="Areas to evaluate (default: A0 A1 A2 A3).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve() if args.outputs_dir else runs_dir
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else outputs_dir / "mt_mrlfads_cross_predict.csv"
    )

    run_dirs = discover_mrlfads_runs(
        runs_dir, run_glob=args.run_glob, recursive=args.recursive
    )

    print(f"Found {len(run_dirs)} MR-LFADS runs under {runs_dir}", flush=True)
    print(f"Outputs dir: {outputs_dir}", flush=True)
    print(f"Areas: {args.area}", flush=True)
    print(f"Metrics: {list(METRICS)}", flush=True)
<<<<<<< Updated upstream
    print(f"R² types: {list(R2_TYPES)}", flush=True)
=======
>>>>>>> Stashed changes
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
