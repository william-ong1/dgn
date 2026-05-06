""" Evaluation metrics for the benchmark to evaluate neural activity reconstruction, communication recovery, and message recovery."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score

from .eval_utils import (
    ArrayMap,
    get_area_names,
    get_holdout_neurons
)


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


def neural_activity_reconstruction(truth: ArrayMap, submission: ArrayMap) -> dict[str, float]:
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
        r2_heldin  = standard_r2(truth[area][:, :, ~mask], submission[area][:, :, ~mask]) if not mask.any() else "n/a"

        results[area_name] = {
            "r2_holdout": r2_holdout,
            "r2_heldin": r2_heldin,
        }

    return results



# --- McFadden metrics disabled (restore when needed) ---
#
# from typing import Literal
#
# import torch
# import torch.nn.functional as F
#
# ReductionType = Literal["mean", "neuron"]
#
#
# def _as_tensor(x: np.ndarray | torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
#     if isinstance(x, torch.Tensor):
#         t = x.float()
#     else:
#         t = torch.from_numpy(np.asarray(x)).float()
#     if device is not None:
#         t = t.to(device)
#     return t
#
#
# def mcfadden_r2_poisson(
#     data: np.ndarray | torch.Tensor,
#     output_params: np.ndarray | torch.Tensor,
#     *,
#     reduction: ReductionType = "mean",
#     null_dims: tuple[int, ...] = (0,),
#     device: torch.device | str | None = None,
# ) -> float | np.ndarray:
#     """
#     McFadden-style R² for Poisson outputs using Poisson NLL (matches MR-LFADS).
#
#     ``output_params`` holds log-firing-rates (same convention as MR-LFADS: ``.exp()`` is passed
#     to ``poisson_nll_loss`` with ``log_input=False``).
#     """
#     data_t = _as_tensor(data, device=device)
#     params = _as_tensor(output_params, device=data_t.device)
#
#     nll_model = F.poisson_nll_loss(
#         input=params.exp(),
#         target=data_t,
#         log_input=False,
#         full=True,
#         reduction="none",
#     )
#
#     output_null = torch.mean(data_t, dim=null_dims, keepdim=True).expand_as(params)
#
#     nll_null = F.poisson_nll_loss(
#         input=output_null,
#         target=data_t,
#         log_input=False,
#         full=True,
#         reduction="none",
#     )
#
#     mask = nll_null > 0
#
#     if reduction == "mean":
#         denom = (nll_null * mask).mean()
#         if torch.isclose(denom, torch.zeros_like(denom)):
#             return float("nan")
#         r2 = 1 - (nll_model * mask).mean() / denom
#         return float(r2.item())
#
#     if reduction == "neuron":
#         denom = (nll_null * mask).mean(dim=(0, 1))
#         num = (nll_model * mask).mean(dim=(0, 1))
#         r2 = 1 - num / denom
#         r2 = torch.where(torch.isclose(denom, torch.zeros_like(denom)), torch.full_like(r2, float("nan")), r2)
#         return r2.detach().cpu().numpy()
#
#     raise ValueError(f"Unknown reduction: {reduction!r}")
#