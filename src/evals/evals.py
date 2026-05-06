from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import Any

from .eval_utils import (
    ArrayMap,
    infer_submission_pred_time_len,
    load_memory_network_lag,
    load_session_arrays,
    slice_truth_for_time_alignment,
)

from .metrics import (
    effectome_cosine_similarity,
    effectome_cosine_similarity_pass_decision,
    lag_recovery_memory_network,
    message_latent_reconstruction,
    message_reconstruction,
    neural_activity_reconstruction,
    truth_input_decoding_memory_network,
    truth_input_decoding_pass_decision,
    message_p_to_d_reconstruction
)


def _safe_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _collect_neural_activity_means(neural_activity: dict[str, dict]) -> dict[str, float]:
    held_out_vals: list[float] = []
    held_in_vals: list[float] = []
    held_out_mcfadden_vals: list[float] = []
    held_in_mcfadden_vals: list[float] = []
    for metrics in neural_activity.values():
        if not isinstance(metrics, dict):
            continue
        v_out = metrics.get("r2-held-out", float("nan"))
        v_in = metrics.get("r2-held-in", float("nan"))
        v_out_m = metrics.get("r2-held-out-mcfadden", float("nan"))
        v_in_m = metrics.get("r2-held-in-mcfadden", float("nan"))
        if isinstance(v_out, (int, float, np.floating)):
            held_out_vals.append(float(v_out))
        if isinstance(v_in, (int, float, np.floating)):
            held_in_vals.append(float(v_in))
        if isinstance(v_out_m, (int, float, np.floating)):
            held_out_mcfadden_vals.append(float(v_out_m))
        if isinstance(v_in_m, (int, float, np.floating)):
            held_in_mcfadden_vals.append(float(v_in_m))

    out = {
        "r2-held-out-mean": _safe_mean(held_out_vals),
        "r2-held-in-mean": _safe_mean(held_in_vals),
    }
    if held_out_mcfadden_vals:
        out["r2-held-out-mcfadden-mean"] = _safe_mean(held_out_mcfadden_vals)
    if held_in_mcfadden_vals:
        out["r2-held-in-mcfadden-mean"] = _safe_mean(held_in_mcfadden_vals)
    return out


def _collect_truth_decode_mean(truth_decode: dict[str, float]) -> dict[str, float]:
    vals: list[float] = []
    for value in truth_decode.values():
        if isinstance(value, (int, float, np.floating)):
            vals.append(float(value))
    return {"truth-inp-decode-r2-mean": _safe_mean(vals)}


def evaluate_submission(
    submission_h5: Path | str,
    truth_h5: Path | str,
    config_dir: Path | str,
    experiment_type: str,
    output_dist: str,
    *,
    truth_time_start: int = 0,
    bootstrap_n: int = 0,
    bootstrap_seed: int = 0,
) -> Any:
    """Evaluate a submission HDF5 against ground truth."""
    submission_path = Path(submission_h5).expanduser().resolve()
    truth_path = Path(truth_h5).expanduser().resolve()
    config_path = Path(config_dir).expanduser().resolve()

    if not config_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_path}")

    submission = load_session_arrays(submission_path)
    truth_full = load_session_arrays(truth_path)

    pred_t = infer_submission_pred_time_len(submission)
    truth = slice_truth_for_time_alignment(truth_full, truth_time_start=truth_time_start, pred_time_len=pred_t)

    if experiment_type == "memory_network":
        results = evaluate_memory_network_submission(submission, truth, config_path, output_dist)
    elif experiment_type == "pass_decision":
        results = evaluate_pass_decision_submission(submission, truth, config_path, output_dist)
    elif experiment_type == "multi_task":
        results = evaluate_multi_task_submission(submission, truth, config_path, output_dist)
    else:
        results = {}

    if bootstrap_n > 0:
        results["confidence_intervals"] = _bootstrap_confidence_intervals(
            submission=submission,
            truth=truth,
            config_dir=config_path,
            experiment_type=experiment_type,
            output_dist=output_dist,
            n_bootstrap=bootstrap_n,
            seed=bootstrap_seed,
        )
    return results


def _bootstrap_confidence_intervals(
    *,
    submission: ArrayMap,
    truth: ArrayMap,
    config_dir: Path,
    experiment_type: str,
    output_dist: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    area_keys = sorted(k for k in submission.keys() if k.startswith("area-"))
    if not area_keys:
        return {}
    n_batch = int(submission[area_keys[0]].shape[0])
    if n_batch <= 1:
        return {}

    rng = np.random.default_rng(seed)
    sampled_values: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_batch, size=n_batch)
        sub_b = _resample_arraymap(submission, idx, n_batch)
        tru_b = _resample_arraymap(truth, idx, n_batch)

        if experiment_type == "memory_network":
            res_b = evaluate_memory_network_submission(sub_b, tru_b, config_dir, output_dist)
        elif experiment_type == "pass_decision":
            res_b = evaluate_pass_decision_submission(sub_b, tru_b, config_dir, output_dist)
        elif experiment_type == "multi_task":
            res_b = evaluate_multi_task_submission(sub_b, tru_b, config_dir, output_dist)
        else:
            continue

        flat = _flatten_numeric_results(res_b)
        for key, value in flat.items():
            sampled_values.setdefault(key, []).append(value)

    ci: dict[str, dict[str, float]] = {}
    for key, values in sampled_values.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            continue
        ci[key] = {
            "low": float(np.nanpercentile(arr, 2.5)),
            "high": float(np.nanpercentile(arr, 97.5)),
        }
    return ci


def _resample_arraymap(arrmap: ArrayMap, idx: np.ndarray, n_batch: int) -> ArrayMap:
    out: ArrayMap = {}
    for key, arr in arrmap.items():
        if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == n_batch:
            out[key] = arr[idx]
        else:
            out[key] = arr
    return out


def _flatten_numeric_results(results: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in results.items():
        if key == "confidence_intervals":
            continue
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_numeric_results(value, flat_key))
        elif isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            out[flat_key] = float(value)
    return out


def evaluate_memory_network_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, output_dist: str) -> Any:
    """Evaluate a memory network submission against ground truth."""

    results = {}

    # Evaluate neural activity reconstruction
    neural_activity_reconstruction_results = neural_activity_reconstruction(submission, truth, output_dist)
    results["neural-activity"] = neural_activity_reconstruction_results

    # Evaluate effectome recovery
    effectome_recovery_results = effectome_cosine_similarity(submission, config_dir)

    # Evaluate message reconstruction
    message_recon_results = message_reconstruction(truth, submission)
    message_latent_recon_results = message_latent_reconstruction(truth, submission)

    # Combine effectome recovery and message reconstruction results
    struct = {}
    struct.update(effectome_recovery_results)
    struct.update(message_recon_results)
    struct.update(message_latent_recon_results)
    results["structure"] = struct

    # Evaluate truth input decoding
    truth_inp_decode_results = truth_input_decoding_memory_network(truth, submission, config_dir)
    results["truth-inp-decode"] = truth_inp_decode_results

    aggregates = {}
    aggregates.update(_collect_neural_activity_means(neural_activity_reconstruction_results))
    aggregates.update(_collect_truth_decode_mean(truth_inp_decode_results))
    results["aggregates"] = aggregates

    # Evaluate temporal lag recovery
    temporal_results = lag_recovery_memory_network(truth, submission)
    true_lag = float(load_memory_network_lag(config_dir))
    temporal_results["lag-true"] = true_lag
    pred_lag = temporal_results.get("lag-pred", float("nan"))
    temporal_results["lag-error"] = abs(pred_lag - true_lag)
    results["temporal"] = temporal_results

    return results


def evaluate_pass_decision_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, output_dist: str) -> Any:
    """Evaluate a pass decision submission against ground truth."""
    
    results = {}

    # Evaluate neural activity reconstruction
    neural_activity_reconstruction_results = neural_activity_reconstruction(submission, truth, output_dist)
    results["neural-activity"] = neural_activity_reconstruction_results

    # Evaluate message reconstruction
    # message_recon_results = message_reconstruction(truth, submission)
    message_p_to_d_recon_results = message_p_to_d_reconstruction(truth, submission)
    effectome_recovery_results = effectome_cosine_similarity_pass_decision(submission)
    # message_latent_recon_results = message_latent_reconstruction(truth, submission)

    # Combine message reconstruction results
    struct = {}
    struct.update(effectome_recovery_results)
    struct.update(message_p_to_d_recon_results)
    # struct.update(message_latent_recon_results)
    results["structure"] = struct

    # Evaluate truth input decoding
    truth_inp_decode_results = truth_input_decoding_pass_decision(truth, submission, config_dir)
    results["truth-inp-decode"] = truth_inp_decode_results

    aggregates = {}
    aggregates.update(_collect_neural_activity_means(neural_activity_reconstruction_results))
    aggregates.update(_collect_truth_decode_mean(truth_inp_decode_results))
    results["aggregates"] = aggregates



    return results


def evaluate_multi_task_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, output_dist: str) -> Any:
    """Evaluate a multi-task submission against ground truth."""

    results = {}
    results.update(neural_activity_reconstruction(submission, truth, output_dist))
    return results