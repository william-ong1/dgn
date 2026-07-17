""" Evaluation metrics to evaluate neural activity reconstruction, connectivity recovery, and message recovery."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


from .eval_utils import (
    ArrayMap,
    get_area_names,
    get_heldout_neurons,
    load_memory_network_connectome_and_ranks,
)


def neural_activity_reconstruction(submission: ArrayMap, truth: ArrayMap, distribution: str) -> dict[str, float]:
    """
    Computes R2 scores for held-out and held-in neuron activity between truth and submission for each area.
    """
    results = {}
    area_names = get_area_names(submission)
    heldout_neurons = get_heldout_neurons(submission)

    for area_name in area_names:
        area = "area-" + area_name
        mask = np.zeros(truth[area].shape[2], dtype=bool)
        mask[heldout_neurons.get(area, [])] = True

        r2_heldout = standard_r2(truth[area][:, :, mask], submission[area][:, :, mask]) if mask.any() else "n/a"
        r2_heldin = standard_r2(truth[area][:, :, ~mask], submission[area][:, :, ~mask]) if not mask.all() else "n/a"

        results[area_name] = {
                "r2-held-out": r2_heldout,
                "r2-held-in": r2_heldin,
        }

        if distribution == "poisson":
            r2_heldout_mcfadden = mcfadden_r2_poisson(truth[area][:, :, mask], submission[area][:, :, mask]) if mask.any() else "n/a"
            r2_heldin_mcfadden = mcfadden_r2_poisson(truth[area][:, :, ~mask], submission[area][:, :, ~mask]) if not mask.all() else "n/a"

            results[area_name].update({
                "r2-held-out-mcfadden": r2_heldout_mcfadden,
                "r2-held-in-mcfadden": r2_heldin_mcfadden,
            })

    return results


def neural_activity_rate_reconstruction(submission: ArrayMap, truth_rates: ArrayMap) -> dict[str, dict[str, float]]:
    """
    Standard R² between predicted and ground-truth Poisson rates (λ per bin).

    ``truth_rates`` should already be in the same units as the submission (expected
    counts per bin), typically derived from continuous activity via ``activity_to_lam``.
    """
    results: dict[str, dict[str, float]] = {}
    area_names = get_area_names(submission)
    heldout_neurons = get_heldout_neurons(submission)

    for area_name in area_names:
        area = "area-" + area_name
        if area not in truth_rates:
            continue

        mask = np.zeros(truth_rates[area].shape[2], dtype=bool)
        mask[heldout_neurons.get(area, [])] = True

        r2_heldout = (
            standard_r2(truth_rates[area][:, :, mask], submission[area][:, :, mask])
            if mask.any()
            else "n/a"
        )
        r2_heldin = (
            standard_r2(truth_rates[area][:, :, ~mask], submission[area][:, :, ~mask])
            if not mask.all()
            else "n/a"
        )

        results[area_name] = {
            "r2-held-out-rates": r2_heldout,
            "r2-held-in-rates": r2_heldin,
        }

    return results


def effectome_cosine_similarity(submission: ArrayMap, config_dir: Path):
    """ 
    Computes cosine similarity for effectome and inferred-input scores between truth and submission.
    """
    results = {}
    true_connectome, true_ranks = load_memory_network_connectome_and_ranks(config_dir)

    if "effectome-scores" in submission:
        pred_effectome_scores = submission.get("effectome-scores", None)
        pred_effectome = np.asarray(pred_effectome_scores, dtype=np.float64)
        np.fill_diagonal(pred_effectome, 0)
        np.fill_diagonal(true_connectome, 0)
        cosine_score = cosine_similarity(pred_effectome, true_connectome * true_ranks.reshape(-1, 1))
        results.update({
            "effectome-cos-sim": cosine_score,
        })
        
    if "inferred-input-scores" in submission:
        pred_input_scores = submission.get("inferred-input-scores", None)
        cosine_score = cosine_similarity(pred_input_scores, true_ranks)
        results.update({
            "inferred-input-cos-sim": cosine_score,
        })

    return results


def effectome_cosine_similarity_pass_decision(submission: ArrayMap) -> dict[str, float]:
    """
    Effectome recovery for pass-decision with fixed ground truth graph: P -> D only.

    Uses submission ``effectome-scores`` if present, binarized at > 0, then computes
    cosine similarity against the fixed adjacency in submission area order.
    """
    if "effectome-scores" not in submission:
        return {}

    area_names = get_area_names(submission)
    if len(area_names) == 0:
        return {}

    pred_scores = np.asarray(submission["effectome-scores"], dtype=np.float64)
    if pred_scores.ndim != 2:
        return {}
    if pred_scores.shape[0] != len(area_names) or pred_scores.shape[1] != len(area_names):
        return {}

    pred_effectome = pred_scores.astype(np.float64)
    np.fill_diagonal(pred_effectome, 0)

    true_effectome = np.zeros_like(pred_effectome, dtype=np.int64)
    p_idx = next((i for i, n in enumerate(area_names) if n.lower().startswith("p")), None)
    d_idx = next((i for i, n in enumerate(area_names) if n.lower().startswith("d")), None)
    if p_idx is None or d_idx is None:
        return {}

    # Pass-decision convention: row=source, col=target (P -> D).
    true_effectome[p_idx, d_idx] = 1
    np.fill_diagonal(true_effectome, 0)

    results = {"effectome-cos-sim": cosine_similarity(pred_effectome, true_effectome)}

    if "inferred-input-scores" in submission:
        pred_input_scores = np.asarray(submission["inferred-input-scores"], dtype=np.float64).reshape(-1)
        if pred_input_scores.shape[0] == len(area_names):
            # Fixed pass-decision convention: D has stronger inferred-input role than P.
            true_input_scores = np.zeros(len(area_names), dtype=np.float64)
            true_input_scores[d_idx] = 1.0
            results["inferred-input-cos-sim"] = cosine_similarity(pred_input_scores, true_input_scores)

    return results


def truth_input_decoding_memory_network(
    truth: ArrayMap,
    submission: ArrayMap,
    config_dir: Path,
    ridge_alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Compute decodability of each region's ground-truth input from the region's hidden activity, returning held out trial-level R² scores.
    """
    _, ranks = load_memory_network_connectome_and_ranks(config_dir)
    area_names = get_area_names(submission)

    # Ground-truth input to all regions
    y_all = np.asarray(truth["truth-inp"], dtype=np.float64)
    n_batch = y_all.shape[0]
    train_idx, test_idx = train_test_split(np.arange(n_batch), test_size=test_size, random_state=random_state, shuffle=True)

    results = {}
    offset = 0

    for i, name in enumerate(area_names):
        r = int(ranks[i])
        sl = slice(offset, offset + r)

        # Ground-truth input to the region
        yi_all = y_all[..., sl]

        # Hidden activity from the region
        xi_all = np.asarray(submission[f"area-{name}"], dtype=np.float64)

        x_train = xi_all[train_idx].reshape(-1, xi_all.shape[-1])
        y_train = yi_all[train_idx].reshape(-1, r)
        x_test = xi_all[test_idx].reshape(-1, xi_all.shape[-1])
        y_test = yi_all[test_idx].reshape(-1, r)

        # Train decoder on held-in trials
        decoder = Ridge(alpha=ridge_alpha)
        decoder.fit(x_train, y_train)

        # Test decoder on held-out trials
        y_pred = decoder.predict(x_test)
        results[name] = float(r2_score(y_test, y_pred, multioutput="uniform_average"))
        offset += r

    return results


def _pass_decision_targets_from_truth_inp(truth: ArrayMap) -> dict[str, np.ndarray]:
    """Build decode targets from ``truth-inp``: raw input, cumsum, and sign(cumsum)."""
    if "truth-inp" not in truth:
        return {}
    inp = np.asarray(truth["truth-inp"], dtype=np.float64)
    cumsum = np.cumsum(inp, axis=1)
    return {
        "truth-inp": inp,
        "cumsum": cumsum,
        "sign_cumsum": np.sign(cumsum),
    }


def truth_input_decoding_pass_decision(
    truth: ArrayMap,
    submission: ArrayMap,
    config_dir: Path | None = None,
) -> dict[str, float | dict[str, float]]:
    """
    Area-specific decodability for pass-decision.

    Multi-area (P+D):
      - P area activity decodes ``truth-inp``
      - D area activity decodes ``message-p_to_d``

    Single-area (P-only or D-only):
      - Decode ``truth-inp``, ``cumsum``, and ``sign_cumsum`` from that area
        (D's primary task signal is ``sign_cumsum`` / ``cumsum``).
    """
    area_names = get_area_names(submission)
    single_area = len(area_names) == 1

    results: dict[str, float | dict[str, float]] = {}
    if single_area:
        targets = _pass_decision_targets_from_truth_inp(truth)
        if not targets:
            return results

        for name in area_names:
            key = f"area-{name}"
            if key not in submission:
                continue
            x = np.asarray(submission[key], dtype=np.float64)
            area_scores: dict[str, float] = {}
            for tname, target in targets.items():
                area_scores[tname] = float(decode_r2_from_features(target, x))
            results[name] = area_scores
        return results

    for name in area_names:
        key = f"area-{name}"
        if key not in submission:
            continue

        lname = name.lower()
        if lname.startswith("p"):
            target_key = "truth-inp"
        elif lname.startswith("d"):
            target_key = "message-p_to_d"
        else:
            continue

        if target_key not in truth:
            continue

        target = np.asarray(truth[target_key], dtype=np.float64)
        x = np.asarray(submission[key], dtype=np.float64)
        score = decode_r2_from_features(target, x)
        results[name] = float(score)

    return results


def message_reconstruction(truth: ArrayMap, submission: ArrayMap) -> dict[str, float]:
    """
    Message reconstruction R² by decoding the truth messages.
    """
    if "message-mesgs" not in submission: return {}
    y = np.asarray(truth["message-mesgs"], dtype=np.float64)
    y_hat = np.asarray(submission["message-mesgs"], dtype=np.float64)
    return {"message-r2": decode_r2_from_features(y, y_hat)}


def message_p_to_d_reconstruction(truth: ArrayMap, submission: ArrayMap) -> dict[str, float]:
    """
    Message p_to_d reconstruction R² by decoding truth `message-p_to_d`.
    """
    if "message-p_to_d" not in submission or "message-p_to_d" not in truth:
        return {}
    y = np.asarray(truth["message-p_to_d"], dtype=np.float64)
    y_hat = np.asarray(submission["message-p_to_d"], dtype=np.float64)
    return {"message-p_to_d-r2": decode_r2_from_features(y, y_hat)}


def message_latent_reconstruction(truth: ArrayMap, submission: ArrayMap) -> dict[str, float]:
    """
    Message latent reconstruction R² by decoding truth `message-latents`.
    """
    if "message-latents" not in submission: return {}
    y = np.asarray(truth["message-latents"], dtype=np.float64)
    y_hat = np.asarray(submission["message-latents"], dtype=np.float64)
    return {"message-latents-r2": decode_r2_from_features(y, y_hat)}


def lag_recovery_memory_network(
    truth: ArrayMap,
    submission: ArrayMap,
    max_lag: int = 5,
    *,
    true_lag: int | None = None,
) -> dict[str, float]:
    """
    Lag scan using model-vs-truth message trajectories directly.

    For lag in [-max_lag, max_lag], align model/truth message tensors in time and score
    decode R^2 (truth decoded from model features). Pick lag with highest R^2.

    Returns:
      - lag-pred: best lag (signed)
      - lag-r2-best: best decode-R² achieved during lag scan
      - lag-r2: decode-R² at ``true_lag`` (when provided), for cross-run comparison
    """

    nan_result = {"lag-pred": float("nan"), "lag-r2-best": float("nan")}
    if true_lag is not None:
        nan_result["lag-r2"] = float("nan")

    if "message-mesgs" not in submission:
        return nan_result

    X = np.asarray(submission["message-mesgs"], dtype=np.float64)
    Y = np.asarray(truth["message-mesgs"], dtype=np.float64)

    best_lag = 0
    best_r2 = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        x_aligned, y_aligned = _align_source_delay(X, Y, lag)
        if x_aligned is None:
            continue

        score = decode_r2_from_features(y_aligned, x_aligned)
        if np.isfinite(score) and score > best_r2:
            best_r2 = score
            best_lag = lag

    if not np.isfinite(best_r2):
        return nan_result

    results = {"lag-pred": float(best_lag), "lag-r2-best": float(best_r2)}

    if true_lag is not None:
        x_aligned, y_aligned = _align_source_delay(X, Y, int(true_lag))
        if x_aligned is None:
            results["lag-r2"] = float("nan")
        else:
            results["lag-r2"] = float(decode_r2_from_features(y_aligned, x_aligned))

    return results


def decode_r2_from_features(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Decode R² from features, splitting into train and test sets and fitting a ridge decoder."""
    n_batch = y.shape[0]
    train_idx, test_idx = train_test_split(np.arange(n_batch), test_size=0.2, random_state=0, shuffle=True)

    x_train = y_hat[train_idx].reshape(-1, y_hat.shape[-1])
    y_train = y[train_idx].reshape(-1, y.shape[-1])
    x_test = y_hat[test_idx].reshape(-1, y_hat.shape[-1])
    y_test = y[test_idx].reshape(-1, y.shape[-1])

    decoder = Ridge(alpha=1.0)
    decoder.fit(x_train, y_train)
    y_pred = decoder.predict(x_test)
    return standard_r2(y_test, y_pred)


def standard_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard coefficient of determination R² on flattened arrays.

    Returns nan if y_true is (near-)constant, matching the usual undefined R² case.
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_t.size != y_p.size:
        raise ValueError(f"Shape mismatch after flatten: {y_t.size} vs {y_p.size}")
    if np.var(y_t) < 1e-12:
        return float("nan")
    return float(r2_score(y_t, y_p))


def mcfadden_r2_poisson(data: np.ndarray, output_params: np.ndarray, *, reduction = "mean", null_dims: tuple[int, ...] = (0,)) -> float:
    """
    McFadden-style R² for Poisson outputs using Poisson NLL.
    """
    data_t = torch.from_numpy(data).float()
    params = torch.from_numpy(output_params).float()

    nll_model = F.poisson_nll_loss(
        input=params,
        target=data_t,
        log_input=False,
        full=True,
        reduction="none",
    )

    output_null = torch.mean(data_t, dim=null_dims, keepdim=True).expand_as(params)

    nll_null = F.poisson_nll_loss(
        input=output_null,
        target=data_t,
        log_input=False,
        full=True,
        reduction="none",
    )

    mask = nll_null > 0

    if reduction == "mean":
        denom = (nll_null * mask).mean()
        if torch.isclose(denom, torch.zeros_like(denom)):
            return float("nan")
        r2 = 1 - (nll_model * mask).mean() / denom
        return float(r2.item())

    if reduction == "neuron":
        denom = (nll_null * mask).mean(dim=(0, 1))
        num = (nll_model * mask).mean(dim=(0, 1))
        r2 = 1 - num / denom
        r2 = torch.where(torch.isclose(denom, torch.zeros_like(denom)), torch.full_like(r2, float("nan")), r2)
        return r2.detach().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two arrays.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
        
    return float(np.dot(a, b) / (na * nb))


def _align_source_delay(X_tgt: np.ndarray, Y_src: np.ndarray, lag: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Align target features X_tgt(t) with delayed source inputs Y_src(t-lag).
    """
    T = min(X_tgt.shape[1], Y_src.shape[1])
    if T <= 1 or lag < 0 or lag >= T:
        return None, None
    Xc = X_tgt[:, :T]
    Yc = Y_src[:, :T]
    if lag == 0:
        return Xc, Yc
    return Xc[:, lag:], Yc[:, :-lag]
