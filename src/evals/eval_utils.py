"""HDF5 loading and time-alignment helpers for evaluation."""

from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import yaml
import pandas as pd
from typing import Any

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


def activity_to_lam(
    activity: np.ndarray,
    *,
    dt: float,
    rate_max: float,
) -> np.ndarray:
    """Map continuous hidden activity to Poisson rate λ (expected counts per bin)."""
    activity = np.clip(np.asarray(activity, dtype=np.float64), -1.0, 1.0)
    rates_hz = rate_max * (activity + 1.0) / 2.0
    lam = rates_hz * dt
    return np.clip(lam, 0.0, None).astype(np.float32)


def continuous_activity_to_rates_map(
    activity_map: ArrayMap,
    *,
    dt: float,
    rate_max: float,
) -> ArrayMap:
    """Convert area-* continuous activity arrays to λ per bin."""
    out: ArrayMap = {}
    for key, arr in activity_map.items():
        if key.startswith("area-"):
            out[key] = activity_to_lam(arr, dt=dt, rate_max=rate_max)
        elif key.startswith("meta-"):
            out[key] = arr
    return out


# Get heldout neurons for each area
def get_heldout_neurons(submission: ArrayMap) -> dict[str, np.ndarray]:
    heldout_neurons = {}
    for k in submission.keys():
        if k.startswith("meta-held-out-neuron-indices"):
            heldout_neurons[k.removeprefix("meta-held-out-neuron-indices-")] = submission[k]
    
    return heldout_neurons


# Load memory-network connectome and ranks from model config
def load_memory_network_connectome_and_ranks(config_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    model_cfg = config_dir / "model" / "model.yaml"

    with model_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    connectome = np.asarray(cfg["connectome"], dtype=np.int64)
    ranks = np.asarray(cfg["ranks"], dtype=np.int64).reshape(-1)

    return connectome, ranks

# Load memory-network lag from model config
def load_memory_network_lag(config_dir: Path) -> int:
    model_cfg = config_dir / "model" / "model.yaml"
    
    with model_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    return int(cfg["lag"])


# Split results for display on the dashboard
def split_results_for_display(results: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_rows: dict[str, dict] = {}
    global_metrics: dict[str, float] = {}

    neural = results.get("neural-activity", {})
    if isinstance(neural, dict):
        for region, metrics in neural.items():
            if isinstance(metrics, dict):
                region_rows.setdefault(region, {}).update(metrics)

    truth_decode = results.get("truth-inp-decode", {})
    if isinstance(truth_decode, dict):
        for region, value in truth_decode.items():
            if isinstance(value, dict):
                for tname, score in value.items():
                    col = "truth-inp-decode-r2" if tname == "truth-inp" else f"{tname}-decode-r2"
                    region_rows.setdefault(region, {})[col] = score
            else:
                region_rows.setdefault(region, {})["truth-inp-decode-r2"] = value

    structure = results.get("structure", {})
    if isinstance(structure, dict):
        global_metrics.update(structure)

    aggregates = results.get("aggregates", {})
    if isinstance(aggregates, dict):
        global_metrics.update(aggregates)

    temporal = results.get("temporal", {})
    temporal_metrics: dict[str, float] = {}
    if isinstance(temporal, dict):
        temporal_metrics.update(temporal)

    # Backward compatibility for any top-level scalar metrics.
    for key, value in results.items():
        if not isinstance(value, dict):
            global_metrics[key] = value

    region_df = pd.DataFrame.from_dict(region_rows, orient="index")
    if not region_df.empty:
        region_df.index.name = "region"
        region_df = region_df.reset_index()

    global_df = pd.DataFrame([global_metrics]) if global_metrics else pd.DataFrame()
    temporal_df = pd.DataFrame([temporal_metrics]) if temporal_metrics else pd.DataFrame()
    return region_df, global_df, temporal_df
