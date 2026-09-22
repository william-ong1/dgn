"""Content-based effectome scoring (Pospisil-style communication volumes).

``M[target, source] = ||m^{source → target}||_2`` over trials, time, and
channels. When the target region's input weights are available, the
dynamically weighted volume is ``||W^{target ← source} m^{source → target}||_2``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .eval_utils import (
    ArrayMap,
    load_memory_network_connectome_and_ranks,
    load_multi_task_spec,
    mt_message_slots,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def l2_volume(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.sum(np.square(x))))


def weighted_l2_volume(weight: np.ndarray, message: np.ndarray) -> float:
    """``||W m||_2`` over trials and time. ``W`` is ``(hidden, channels)``."""
    w = np.asarray(weight, dtype=np.float64)
    m = np.asarray(message, dtype=np.float64)
    if m.ndim != 3:
        raise ValueError(f"Expected message (batch, time, channels), got {m.shape}")
    if w.ndim != 2 or w.shape[1] != m.shape[-1]:
        raise ValueError(f"Weight {w.shape} incompatible with message {m.shape}")
    contrib = np.einsum("btc,hc->bth", m, w)
    return l2_volume(contrib)


def zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(out, 0.0)
    return out


def mn_message_slots(
    connectome: np.ndarray, ranks: np.ndarray
) -> list[dict[str, Any]]:
    """Flattened ``message-mesgs`` slots: row = target, column = source."""
    n = len(ranks)
    effectome = np.tile(ranks.reshape(1, -1), (n, 1)) * np.asarray(connectome)
    slots: list[dict[str, Any]] = []
    offset = 0
    for ti in range(n):
        for sj in range(n):
            width = int(effectome[ti, sj])
            if width <= 0:
                continue
            slots.append(
                {
                    "target": f"A{ti}",
                    "source": f"A{sj}",
                    "target_idx": ti,
                    "source_idx": sj,
                    "is_self": ti == sj,
                    "slice": slice(offset, offset + width),
                }
            )
            offset += width
    return slots


def effectome_from_mn_messages(
    messages: np.ndarray,
    connectome: np.ndarray,
    ranks: np.ndarray,
    *,
    incoming_weights: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """L2 (and optional weighted L2) effectome from DGN ``message-mesgs``."""
    n = len(ranks)
    volume = np.zeros((n, n), dtype=np.float64)
    weighted = np.zeros((n, n), dtype=np.float64) if incoming_weights else None
    slots = mn_message_slots(connectome, ranks)
    expected = slots[-1]["slice"].stop if slots else 0
    if messages.shape[-1] < expected:
        raise ValueError(
            f"message-mesgs has {messages.shape[-1]} channels, expected >= {expected}"
        )

    for slot in slots:
        if slot["is_self"]:
            continue
        block = messages[..., slot["slice"]]
        ti, sj = slot["target_idx"], slot["source_idx"]
        volume[ti, sj] = l2_volume(block)
        if weighted is not None:
            key = (slot["target"], slot["source"])
            w = incoming_weights.get(key)
            if w is not None and w.size and w.shape[1] == block.shape[-1]:
                weighted[ti, sj] = weighted_l2_volume(w, block)
            else:
                weighted[ti, sj] = volume[ti, sj]
    return volume, weighted


def inferred_input_from_truth_inp(truth_inp: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Per-area L2 of the ground-truth inferred-input channels."""
    scores = np.zeros(len(ranks), dtype=np.float64)
    offset = 0
    for i, rank in enumerate(ranks):
        width = int(rank)
        scores[i] = l2_volume(truth_inp[..., offset : offset + width])
        offset += width
    return scores


def effectome_from_pd_messages(
    p_to_d: np.ndarray,
    area_names: list[str],
    *,
    incoming_weights: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Single true edge P→D from ``message-p_to_d``."""
    n = len(area_names)
    volume = np.zeros((n, n), dtype=np.float64)
    weighted = np.zeros((n, n), dtype=np.float64) if incoming_weights else None
    try:
        p_idx = next(i for i, name in enumerate(area_names) if name.lower().startswith("p"))
        d_idx = next(i for i, name in enumerate(area_names) if name.lower().startswith("d"))
    except StopIteration as exc:
        raise ValueError(f"Need P and D in area names, got {area_names}") from exc

    volume[d_idx, p_idx] = l2_volume(p_to_d)
    if weighted is not None:
        w = incoming_weights.get(("D", "P"))
        if w is None:
            w = incoming_weights.get((area_names[d_idx], area_names[p_idx]))
        if w is not None and w.size and w.shape[1] == p_to_d.shape[-1]:
            weighted[d_idx, p_idx] = weighted_l2_volume(w, p_to_d)
        else:
            weighted[d_idx, p_idx] = volume[d_idx, p_idx]
    return volume, weighted


def _score_pair(pred: np.ndarray, true: np.ndarray) -> float:
    return _cosine(zero_diagonal(pred), zero_diagonal(true))


def _unit(vec: np.ndarray) -> np.ndarray | None:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 0.0:
        return None
    return x / n


def combined_structure_cosine(
    pred_messages: np.ndarray | None,
    true_messages: np.ndarray | None,
    pred_inputs: np.ndarray | None,
    true_inputs: np.ndarray | None,
) -> float:
    """Cosine of concatenated, separately unit-normalized message and input volumes.

    Equal weight on inferred messages and inferred inputs (the raw L2 magnitudes
    do not let one block dominate). Equals the mean of the two block cosines
    when both are defined.
    """
    parts_pred: list[np.ndarray] = []
    parts_true: list[np.ndarray] = []

    if pred_messages is not None and true_messages is not None:
        pm = _unit(zero_diagonal(pred_messages))
        tm = _unit(zero_diagonal(true_messages))
        if pm is not None and tm is not None and pm.shape == tm.shape:
            parts_pred.append(pm)
            parts_true.append(tm)

    if pred_inputs is not None and true_inputs is not None:
        pu = _unit(pred_inputs)
        tu = _unit(true_inputs)
        if pu is not None and tu is not None and pu.shape == tu.shape:
            parts_pred.append(pu)
            parts_true.append(tu)

    if not parts_pred:
        return float("nan")
    return _cosine(np.concatenate(parts_pred), np.concatenate(parts_true))


def score_memory_network_effectome(
    submission: ArrayMap,
    truth: ArrayMap | None,
    config_dir: Path,
    *,
    dgn_run_dir: Path | str | None = None,
    mrlfads_run_dir: Path | str | None = None,
) -> dict[str, float]:
    connectome, ranks = load_memory_network_connectome_and_ranks(config_dir)
    area_names = [f"A{i}" for i in range(len(ranks))]
    results: dict[str, float] = {}

    true_volume = None
    true_weighted = None
    if truth is not None and "message-mesgs" in truth:
        dgn_w = None
        if dgn_run_dir is not None:
            from .transform_matrices import extract_dgn_incoming_weights

            dgn_w = extract_dgn_incoming_weights(Path(dgn_run_dir), "memory_network")
        true_volume, true_weighted = effectome_from_mn_messages(
            np.asarray(truth["message-mesgs"]),
            connectome,
            ranks,
            incoming_weights=dgn_w,
        )

    if true_volume is None:
        # Fallback only when message tensors are missing.
        true_volume = connectome.astype(np.float64) * ranks.reshape(1, -1)

    pred_m = None
    if "effectome-scores" in submission:
        pred = np.asarray(submission["effectome-scores"], dtype=np.float64)
        if pred.shape == true_volume.shape:
            pred_m = pred
            results["effectome-cos-sim"] = _score_pair(pred, true_volume)

        if true_weighted is not None and mrlfads_run_dir is not None and "message-mesgs" in submission:
            pred_weighted = _mrl_weighted_from_mesgs(
                np.asarray(submission["message-mesgs"]),
                Path(mrlfads_run_dir),
                area_names,
            )
            if pred_weighted is not None:
                results["effectome-weighted-cos-sim"] = _score_pair(
                    pred_weighted, true_weighted
                )

    pred_u = None
    true_u = None
    if "inferred-input-scores" in submission and truth is not None and "truth-inp" in truth:
        pred_u = np.asarray(submission["inferred-input-scores"], dtype=np.float64).reshape(-1)
        true_u = inferred_input_from_truth_inp(np.asarray(truth["truth-inp"]), ranks)
        if pred_u.shape == true_u.shape:
            results["inferred-input-cos-sim"] = _cosine(pred_u, true_u)
        else:
            pred_u, true_u = None, None
    elif "inferred-input-scores" in submission:
        pred_u = np.asarray(submission["inferred-input-scores"], dtype=np.float64).reshape(-1)
        if pred_u.shape[0] == len(ranks):
            true_u = ranks.astype(np.float64)
            results["inferred-input-cos-sim"] = _cosine(pred_u, true_u)
        else:
            pred_u = None

    combined = combined_structure_cosine(pred_m, true_volume, pred_u, true_u)
    if np.isfinite(combined):
        results["structure-cos-sim"] = combined

    return results


def effectome_from_mt_messages(
    messages: np.ndarray,
    spec: dict[str, Any],
    *,
    incoming_weights: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """L2 (and optional weighted L2) effectome from MultiTaskNet ``message-mesgs``."""
    n = int(spec["num_areas"])
    volume = np.zeros((n, n), dtype=np.float64)
    weighted = np.zeros((n, n), dtype=np.float64) if incoming_weights else None
    slots = mt_message_slots(spec)
    expected = slots[-1]["slice"].stop if slots else 0
    if messages.shape[-1] < expected:
        raise ValueError(
            f"message-mesgs has {messages.shape[-1]} channels, expected >= {expected}"
        )

    for slot in slots:
        if slot["is_output"] or slot["is_self"]:
            continue
        block = messages[..., slot["slice"]]
        ti, sj = slot["target_idx"], slot["source_idx"]
        volume[ti, sj] = l2_volume(block)
        if weighted is not None:
            w = incoming_weights.get((slot["target"], slot["source"]))
            if w is not None and w.size and w.shape[1] == block.shape[-1]:
                weighted[ti, sj] = weighted_l2_volume(w, block)
            else:
                weighted[ti, sj] = volume[ti, sj]
    return volume, weighted


def score_multi_task_effectome(
    submission: ArrayMap,
    truth: ArrayMap | None,
    config_dir: Path,
    *,
    dgn_run_dir: Path | str | None = None,
    mrlfads_run_dir: Path | str | None = None,
) -> dict[str, float]:
    spec = load_multi_task_spec(config_dir)
    area_names = spec["area_names"]
    n = len(area_names)
    results: dict[str, float] = {}

    true_volume = None
    true_weighted = None
    if truth is not None and "message-mesgs" in truth:
        dgn_w = None
        if dgn_run_dir is not None:
            from .transform_matrices import extract_dgn_incoming_weights

            try:
                dgn_w = extract_dgn_incoming_weights(Path(dgn_run_dir), "multi_task")
            except (FileNotFoundError, KeyError, ValueError):
                dgn_w = None
        try:
            true_volume, true_weighted = effectome_from_mt_messages(
                np.asarray(truth["message-mesgs"]),
                spec,
                incoming_weights=dgn_w,
            )
        except ValueError:
            true_volume = None
            true_weighted = None

    if true_volume is None:
        true_volume = np.zeros((n, n), dtype=np.float64)
        for slot in mt_message_slots(spec):
            if slot["is_output"]:
                continue
            true_volume[slot["target_idx"], slot["source_idx"]] = 1.0

    pred_m = None
    if "effectome-scores" in submission:
        pred = np.asarray(submission["effectome-scores"], dtype=np.float64)
        if pred.shape == true_volume.shape:
            pred_m = pred
            results["effectome-cos-sim"] = _score_pair(pred, true_volume)

        if (
            true_weighted is not None
            and mrlfads_run_dir is not None
            and "message-mesgs" in submission
        ):
            pred_weighted = _mrl_weighted_from_mesgs(
                np.asarray(submission["message-mesgs"]),
                Path(mrlfads_run_dir),
                area_names,
            )
            if pred_weighted is not None:
                results["effectome-weighted-cos-sim"] = _score_pair(
                    pred_weighted, true_weighted
                )

    pred_u = None
    true_u = None
    if "inferred-input-scores" in submission:
        pred_u = np.asarray(submission["inferred-input-scores"], dtype=np.float64).reshape(-1)
        if truth is not None:
            vols = []
            for name in area_names:
                key = f"inputs-{name}"
                vols.append(l2_volume(truth[key]) if key in truth else 0.0)
            true_u = np.asarray(vols, dtype=np.float64)
            if pred_u.shape == true_u.shape:
                results["inferred-input-cos-sim"] = _cosine(pred_u, true_u)
            else:
                pred_u, true_u = None, None
        else:
            pred_u = None

    combined = combined_structure_cosine(pred_m, true_volume, pred_u, true_u)
    if np.isfinite(combined):
        results["structure-cos-sim"] = combined
    return results


def score_pass_decision_effectome(
    submission: ArrayMap,
    truth: ArrayMap | None,
    *,
    dgn_run_dir: Path | str | None = None,
    mrlfads_run_dir: Path | str | None = None,
) -> dict[str, float]:
    # Model / score-matrix order is P then D, not HDF5 key order.
    area_names = ["P", "D"]
    if "effectome-scores" not in submission:
        return {}
    results: dict[str, float] = {}

    true_volume = None
    true_weighted = None
    if truth is not None and "message-p_to_d" in truth:
        dgn_w = None
        if dgn_run_dir is not None:
            from .transform_matrices import extract_dgn_incoming_weights

            dgn_w = extract_dgn_incoming_weights(Path(dgn_run_dir), "pass_decision")
        true_volume, true_weighted = effectome_from_pd_messages(
            np.asarray(truth["message-p_to_d"]),
            area_names,
            incoming_weights=dgn_w,
        )
    if true_volume is None:
        true_volume = np.zeros((2, 2), dtype=np.float64)
        true_volume[1, 0] = 1.0  # target D, source P

    pred_m = None
    if "effectome-scores" in submission:
        pred = np.asarray(submission["effectome-scores"], dtype=np.float64)
        if pred.shape == (2, 2):
            pred_m = pred
            results["effectome-cos-sim"] = _score_pair(pred, true_volume)
        # Weighted pred needs every inferred edge; PD exports only P→D, so a
        # one-hot vs one-hot score would be trivially 1 and is omitted.

    pred_u = None
    true_u = None
    if "inferred-input-scores" in submission:
        cand_u = np.asarray(submission["inferred-input-scores"], dtype=np.float64).reshape(-1)
        if cand_u.shape[0] == 2:
            pred_u = cand_u
            if truth is not None and "truth-inp" in truth:
                true_u = np.array(
                    [l2_volume(np.asarray(truth["truth-inp"])), 0.0], dtype=np.float64
                )
            else:
                true_u = np.array([1.0, 0.0], dtype=np.float64)
            results["inferred-input-cos-sim"] = _cosine(pred_u, true_u)

    combined = combined_structure_cosine(pred_m, true_volume, pred_u, true_u)
    if np.isfinite(combined):
        results["structure-cos-sim"] = combined

    return results


def _mrl_weighted_from_mesgs(
    messages: np.ndarray,
    run_dir: Path,
    area_names: list[str],
    single_edge: tuple[str, str] | None = None,
) -> np.ndarray | None:
    from .transform_matrices import extract_mrl_incoming_weights

    try:
        blocks, model_names, com_dims = extract_mrl_incoming_weights(run_dir)
    except (FileNotFoundError, KeyError, ValueError):
        return None

    n = len(area_names)
    weighted = np.zeros((n, n), dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(area_names)}

    if single_edge is not None:
        tgt, src = single_edge
        w = blocks.get((tgt, src))
        if w is None or not w.size:
            return None
        # Exported P→D message may be com_dim (means) or 2*com_dim; use matching cols.
        width = min(w.shape[1], messages.shape[-1])
        weighted[name_to_idx[tgt], name_to_idx[src]] = weighted_l2_volume(
            w[:, :width], messages[..., :width]
        )
        return weighted

    offset = 0
    for tgt in model_names:
        com_dim = int(com_dims[tgt])
        for src in model_names:
            if src == tgt:
                continue
            block = messages[..., offset : offset + com_dim]
            offset += com_dim
            if tgt not in name_to_idx or src not in name_to_idx:
                continue
            w = blocks.get((tgt, src))
            if w is None or w.shape[1] != com_dim or block.shape[-1] != com_dim:
                continue
            weighted[name_to_idx[tgt], name_to_idx[src]] = weighted_l2_volume(w, block)
    return weighted
