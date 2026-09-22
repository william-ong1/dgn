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
    load_multi_task_spec,
    mrl_message_slots,
    mt_message_slots,
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
        if area not in truth:
            continue
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


def effectome_cosine_similarity(
    submission: ArrayMap,
    config_dir: Path,
    truth: ArrayMap | None = None,
    *,
    dgn_run_dir: Path | str | None = None,
    mrlfads_run_dir: Path | str | None = None,
):
    """
    Effectome cosine similarity from actual DGN message / input content.

    ``M_ij = ||m^{j → i}||_2`` over trials, time, and channels. When DGN input
    weights are available, also reports the dynamically weighted effectome.
    Falls back to the config graph only if ``message-mesgs`` is missing.
    """
    from .effectome import score_memory_network_effectome

    return score_memory_network_effectome(
        submission,
        truth,
        config_dir,
        dgn_run_dir=dgn_run_dir,
        mrlfads_run_dir=mrlfads_run_dir,
    )


def effectome_cosine_similarity_multi_task(
    submission: ArrayMap,
    config_dir: Path,
    truth: ArrayMap | None = None,
    *,
    dgn_run_dir: Path | str | None = None,
    mrlfads_run_dir: Path | str | None = None,
) -> dict[str, float]:
    """
    Effectome cosine similarity for multi-region task-trained DGNs.

    ``M_ij = ||m^{j → i}||_2`` over trials, time, and channels. When DGN input
    weights are available, also reports the dynamically weighted effectome.
    """
    from .effectome import score_multi_task_effectome

    return score_multi_task_effectome(
        submission,
        truth,
        config_dir,
        dgn_run_dir=dgn_run_dir,
        mrlfads_run_dir=mrlfads_run_dir,
    )


def effectome_cosine_similarity_pass_decision(
    submission: ArrayMap,
    truth: ArrayMap | None = None,
    *,
    dgn_run_dir: Path | str | None = None,
    mrlfads_run_dir: Path | str | None = None,
) -> dict[str, float]:
    """
    Pass-decision effectome from actual P→D message content (target × source).

    Inferred-input scores are compared to the L2 of ``truth-inp`` on P (D is 0).
    """
    from .effectome import score_pass_decision_effectome

    return score_pass_decision_effectome(
        submission,
        truth,
        dgn_run_dir=dgn_run_dir,
        mrlfads_run_dir=mrlfads_run_dir,
    )


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

    For every area (P-only, D-only, or P+D), decode ``truth-inp``, ``cumsum``,
    and ``sign_cumsum`` from that area's activity.

    When ``message-p_to_d`` is present in truth (typical multi-area runs), also
    report how well D activity decodes that message channel.

    When ``message-p_to_d`` is present in the submission (MR-LFADS forward export),
    also decode ``cumsum`` from that P→D message itself (expected to be low:
    the message should carry momentary evidence, not the integrated evidence).
    """
    area_names = get_area_names(submission)
    targets = _pass_decision_targets_from_truth_inp(truth)
    results: dict[str, float | dict[str, float]] = {}
    if not targets:
        return results

    msg_target = (
        np.asarray(truth["message-p_to_d"], dtype=np.float64)
        if "message-p_to_d" in truth
        else None
    )

    for name in area_names:
        key = f"area-{name}"
        if key not in submission:
            continue
        x = np.asarray(submission[key], dtype=np.float64)
        area_scores: dict[str, float] = {
            tname: float(decode_r2_from_features(target, x))
            for tname, target in targets.items()
        }
        if msg_target is not None and str(name).lower().startswith("d"):
            area_scores["message-p_to_d"] = float(
                decode_r2_from_features(msg_target, x)
            )
        results[name] = area_scores

    # Predicted P→D message → cumsum (and sibling targets for context)
    if "message-p_to_d" in submission:
        msg = np.asarray(submission["message-p_to_d"], dtype=np.float64)
        results["message-p_to_d"] = {
            tname: float(decode_r2_from_features(target, msg))
            for tname, target in targets.items()
        }

    return results


def truth_input_decoding_multi_task(
    truth: ArrayMap,
    submission: ArrayMap,
) -> dict[str, dict[str, float]]:
    """
    Decode each region's ground-truth external inputs.

    Prefers MR-LFADS inferred inputs (``inferred-input-{area}``) when present,
    and also reports decoding from inferred regional activity (``area-*``).
    """
    area_names = get_area_names(submission)
    results: dict[str, dict[str, float]] = {}
    for name in area_names:
        y = truth.get(f"inputs-{name}")
        if y is None:
            continue
        y = np.asarray(y, dtype=np.float64)
        if y.ndim < 2 or y.shape[-1] == 0 or y.size == 0:
            continue
        area_scores: dict[str, float] = {}
        inferred_key = f"inferred-input-{name}"
        if inferred_key in submission:
            x_inf = np.asarray(submission[inferred_key], dtype=np.float64)
            if x_inf.size and x_inf.shape[-1] > 0:
                area_scores["inferred-input"] = float(decode_r2_from_features(y, x_inf))
        area_key = f"area-{name}"
        if area_key in submission:
            x_act = np.asarray(submission[area_key], dtype=np.float64)
            if x_act.size and x_act.shape[-1] > 0:
                area_scores["truth-inp"] = float(decode_r2_from_features(y, x_act))
        if area_scores:
            results[name] = area_scores
    return results


def message_reconstruction_by_pathway(
    truth: ArrayMap,
    submission: ArrayMap,
    config_dir: Path,
) -> dict[str, float]:
    """
    Per-pathway ridge decode: inferred ``m̂^{j→i}`` → ground-truth ``m^{j→i}``.
    """
    if "message-mesgs" not in submission or "message-mesgs" not in truth:
        return {}
    try:
        spec = load_multi_task_spec(config_dir)
    except FileNotFoundError:
        return {}

    y = np.asarray(truth["message-mesgs"], dtype=np.float64)
    y_hat = np.asarray(submission["message-mesgs"], dtype=np.float64)
    slots = [s for s in mt_message_slots(spec) if not s["is_output"]]
    if not slots:
        return {}

    area_names = spec["area_names"]
    n = len(area_names)
    n_off = n * (n - 1)
    inferred_map: dict[tuple[str, str], slice] = {}
    if n_off > 0 and y_hat.shape[-1] % n_off == 0:
        com_dim = int(y_hat.shape[-1] // n_off)
        for slot in mrl_message_slots(area_names, com_dim):
            inferred_map[(slot["target"], slot["source"])] = slot["slice"]

    results: dict[str, float] = {}
    scores: list[float] = []
    for slot in slots:
        yt = y[..., slot["slice"]]
        if yt.size == 0 or yt.shape[-1] == 0:
            continue
        key = (slot["target"], slot["source"])
        yh = y_hat[..., inferred_map[key]] if key in inferred_map else y_hat
        score = float(decode_r2_from_features(yt, yh))
        results[f"message-r2-{slot['source']}-to-{slot['target']}"] = score
        scores.append(score)
    if scores:
        results["message-r2-pathway-mean"] = float(np.nanmean(np.asarray(scores, dtype=np.float64)))
    return results


def message_reconstruction(truth: ArrayMap, submission: ArrayMap) -> dict[str, float]:
    """
    Message reconstruction R²: linearly predict ground-truth messages from inferred
    messages (Experiment 3, Fig. 4d left).
    """
    if "message-mesgs" not in submission or "message-mesgs" not in truth:
        return {}
    y = np.asarray(truth["message-mesgs"], dtype=np.float64)
    y_hat = np.asarray(submission["message-mesgs"], dtype=np.float64)
    return {"message-r2": decode_r2_from_features(y, y_hat)}


def message_reconstruction_reverse(truth: ArrayMap, submission: ArrayMap) -> dict[str, float]:
    """
    Reverse message reconstruction R²: linearly predict inferred messages from ground-
    truth messages (Experiment 3, Fig. 4d right). Lower scores suggest inferred
    messages carry information beyond the ground truth.
    """
    if "message-mesgs" not in submission or "message-mesgs" not in truth:
        return {}
    y = np.asarray(submission["message-mesgs"], dtype=np.float64)
    y_hat = np.asarray(truth["message-mesgs"], dtype=np.float64)
    return {"message-reverse-r2": decode_r2_from_features(y, y_hat)}


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
        nan_result["lag-error"] = float("nan")

    if "message-mesgs" not in submission or "message-mesgs" not in truth:
        return nan_result

    X = np.asarray(submission["message-mesgs"], dtype=np.float64)
    Y = np.asarray(truth["message-mesgs"], dtype=np.float64)
    if true_lag is not None:
        max_lag = max(int(max_lag), abs(int(true_lag)) + 3)

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
        results["lag-true"] = float(true_lag)
        results["lag-error"] = float(abs(best_lag - int(true_lag)))
        x_aligned, y_aligned = _align_source_delay(X, Y, int(true_lag))
        if x_aligned is None:
            results["lag-r2"] = float("nan")
        else:
            results["lag-r2"] = float(decode_r2_from_features(y_aligned, x_aligned))
        for delta, tag in ((-2, "m2"), (-1, "m1"), (0, "0"), (1, "p1"), (2, "p2")):
            k = int(true_lag) + delta
            xa, ya = _align_source_delay(X, Y, k)
            results[f"lag-r2-rel-{tag}"] = (
                float("nan")
                if xa is None
                else float(decode_r2_from_features(ya, xa))
            )

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
    Also returns nan if either array contains non-finite values (e.g. diverged preds).
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_t.size != y_p.size:
        raise ValueError(f"Shape mismatch after flatten: {y_t.size} vs {y_p.size}")
    if not (np.isfinite(y_t).all() and np.isfinite(y_p).all()):
        return float("nan")
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
    Align inferred features ``X(t)`` with ground truth ``Y(t - lag)``.

    ``lag > 0``: inferred is later than truth (``X[:, lag:]`` vs ``Y[:, :-lag]``).
    ``lag < 0``: inferred is earlier than truth (``X[:, :T+lag]`` vs ``Y[:, -lag:]``).
    """
    T = min(int(X_tgt.shape[1]), int(Y_src.shape[1]))
    if T <= 1 or abs(int(lag)) >= T:
        return None, None
    Xc = X_tgt[:, :T]
    Yc = Y_src[:, :T]
    lag = int(lag)
    if lag == 0:
        return Xc, Yc
    if lag > 0:
        return Xc[:, lag:], Yc[:, :-lag]
    return Xc[:, : T + lag], Yc[:, -lag:]
