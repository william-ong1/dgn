"""HDF5 loading and time-alignment helpers for evaluation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# ArrayMap is a dictionary of HDF5 dataset names (e.g. "area-A0") and their corresponding numpy arrays.
ArrayMap = dict[str, np.ndarray]

# Load the session arrays from the HDF5 file
def load_session_arrays(h5_path: Path, *, session: str = "0") -> ArrayMap:
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    try:
        with h5py.File(h5_path, "r") as h5:
            if session not in h5:
                raise KeyError(f"Session {session!r} not found in {h5_path}")
            group = h5[session]
            return {key: np.asarray(group[key][:]) for key in group.keys()}
    except OSError as exc:
        raise RuntimeError(f"Could not read HDF5 file {h5_path}: {exc}") from exc


# Helper function to infer the prediction time length from the submission
def infer_submission_pred_time_len(submission: ArrayMap) -> int:
    area_keys = sorted(k for k in submission if k.startswith("area-"))
    if not area_keys:
        raise ValueError("Submission has no area-* datasets; cannot infer prediction time length.")
    # Get the prediction time length from the first area key
    t = submission[area_keys[0]].shape[1]
    for k in area_keys[1:]:
        if submission[k].shape[1] != t:
            raise ValueError(
                f"Inconsistent prediction time length: {area_keys[0]} has T={t}, {k} has T={submission[k].shape[1]}"
            )
    return int(t)


# Slice the truth arrays along time dimension so truth arrays align with the submission time 0, skips meta-* keys.
def slice_truth_for_time_alignment(truth: ArrayMap, *, truth_time_start: int, pred_time_len: int) -> ArrayMap:
    if truth_time_start < 0:
        raise ValueError(f"truth_time_start must be >= 0, got {truth_time_start}")

    end = truth_time_start + pred_time_len
    out: ArrayMap = {}

    # Iterate over the truth arrays and slice them along the time dimension
    for key, arr in truth.items():
        if key.startswith("meta-"):
            out[key] = arr
            continue
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            if arr.shape[1] < end:
                raise ValueError(
                    f"Truth dataset {key!r} has time dim {arr.shape[1]} < {end} "
                    f"(truth_time_start={truth_time_start} + pred_time_len={pred_time_len})."
                )
            out[key] = arr[:, truth_time_start:end]
        else:
            out[key] = arr

    return out


# Get area names from submission
def get_area_names(submission: ArrayMap) -> list[str]:
    return [k.removeprefix("area-") for k in submission.keys() if k.startswith("area-")]


# Get holdout neurons for each area
def get_holdout_neurons(submission: ArrayMap) -> dict[str, np.ndarray]:
    holdout_neurons = {}
    for k in submission.keys():
        if k.startswith("meta-held-out-neuron-indices"):
            holdout_neurons[k.removeprefix("meta-held-out-neuron-indices-")] = submission[k]
    
    return holdout_neurons