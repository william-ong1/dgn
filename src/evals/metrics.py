""" Evaluation metrics to evaluate neural activity reconstruction, communication recovery, and message recovery."""

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
    get_holdout_neurons,
    load_memory_network_connectome_and_ranks,
)


def neural_activity_reconstruction(submission: ArrayMap, truth: ArrayMap, distribution: str) -> dict[str, float]:
    """
    Computes R2 scores for held-out and held-in neuron activity between truth and submission for each area.
    """
    results = {}
    area_names = get_area_names(submission)
    holdout_neurons = get_holdout_neurons(submission)

    for area_name in area_names:
        area = "area-" + area_name
        mask = np.zeros(truth[area].shape[2], dtype=bool)
        mask[holdout_neurons.get(area, [])] = True

        r2_holdout = standard_r2(truth[area][:, :, mask], submission[area][:, :, mask]) if mask.any() else "n/a"
        r2_heldin = standard_r2(truth[area][:, :, ~mask], submission[area][:, :, ~mask]) if not mask.all() else "n/a"

        results[area_name] = {
                "r2-holdout": r2_holdout,
                "r2-heldin": r2_heldin,
        }

        if distribution == "poisson":
            r2_holdout_mcfadden = mcfadden_r2_poisson(truth[area][:, :, mask], submission[area][:, :, mask]) if mask.any() else "n/a"
            r2_heldin_mcfadden = mcfadden_r2_poisson(truth[area][:, :, ~mask], submission[area][:, :, ~mask]) if not mask.all() else "n/a"

        
            results[area_name].update({
                "r2_holdout_mcfadden": r2_holdout_mcfadden,
                "r2_heldin_mcfadden": r2_heldin_mcfadden,
            })

    return results


def effectome_cosine_similarity(submission: ArrayMap, config_dir: Path):
    """ 
    Computes cosine similarity for effectome and inferred-input scores between truth and submission.
    """
    results = {}
    true_connectome, true_ranks = load_memory_network_connectome_and_ranks(config_dir)

    if "effectome-scores" in submission:
        pred_effectome_scores = submission.get("effectome-scores", None)
        pred_effectome = (np.asarray(pred_effectome_scores, dtype=np.float64) > 0).astype(np.int64)
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


def truth_input_decoding_r2(
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
