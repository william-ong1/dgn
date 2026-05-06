from __future__ import annotations
from pathlib import Path
from typing import Any
import h5py
import numpy as np

# Keys are HDF5 dataset names (e.g. "area-A0"); values are numpy arrays.
ArrayMap = dict[str, np.ndarray]


def _load_session_arrays(h5_path: Path, *, session: str = "0") -> ArrayMap:
    """Load every dataset under ``/<session>/`` into memory."""
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


def evaluate_submission(submission_h5: Path | str, truth_h5: Path | str, config_dir: Path | str, experiment_type: str, observation_type: str) -> Any:
    """Evaluate a submission HDF5 against ground truth."""
    submission_path = Path(submission_h5).expanduser().resolve()
    truth_path = Path(truth_h5).expanduser().resolve()
    config_path = Path(config_dir).expanduser().resolve()

    if not config_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_path}")

    submission = _load_session_arrays(submission_path)
    truth = _load_session_arrays(truth_path)

    if experiment_type == "memory_network":
        return evaluate_memory_network_submission(submission, truth, config_path, observation_type)
    if experiment_type == "pass_decision":
        return evaluate_pass_decision_submission(submission, truth, config_path, observation_type)
    if experiment_type == "multi_task":
        return evaluate_multi_task_submission(submission, truth, config_path, observation_type)
    raise ValueError(f"Invalid experiment type: {experiment_type}")


def evaluate_memory_network_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, observation_type: str) -> Any:
    """Evaluate a memory network submission against ground truth."""
    
    # print(submission["area-A0"].shape)
    print(submission.keys())
    print(truth.keys())
    # print(truth["meta-batch-indices"])


def evaluate_pass_decision_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, observation_type: str) -> Any:
    """Evaluate a pass decision submission against ground truth."""
    pass


def evaluate_multi_task_submission(submission: ArrayMap, truth: ArrayMap, config_dir: Path, observation_type: str) -> Any:
    """Evaluate a multi-task submission against ground truth."""
    pass
