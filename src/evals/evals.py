from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import Any

from .eval_utils import (
    ArrayMap,
    infer_submission_pred_time_len,
    load_session_arrays,
    slice_truth_for_time_alignment,
    get_area_names,
    get_holdout_neurons
)

from .metrics import neural_activity_reconstruction, standard_r2


def evaluate_submission(
    submission_h5: Path | str,
    truth_h5: Path | str,
    config_dir: Path | str,
    experiment_type: str,
    distribution: str,
    *,
    truth_time_start: int = 0,
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
        return evaluate_memory_network_submission(submission, truth, config_path, distribution)
    if experiment_type == "pass_decision":
        return evaluate_pass_decision_submission(submission, truth, config_path, distribution)
    if experiment_type == "multi_task":
        return evaluate_multi_task_submission(submission, truth, config_path, distribution)


def evaluate_memory_network_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, distribution: str) -> Any:
    """Evaluate a memory network submission against ground truth."""
    # spec = load_memory_network_spec(config_dir)
    
    results = {}
    results.update(neural_activity_reconstruction(submission, truth))

    return results


def evaluate_pass_decision_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, distribution: str) -> Any:
    """Evaluate a pass decision submission against ground truth."""
    
    results = {}
    results.update(neural_activity_reconstruction(submission, truth))
    return results


def evaluate_multi_task_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, distribution: str) -> Any:
    """Evaluate a multi-task submission against ground truth."""
    pass
