from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils.common_utils import PolyRegression

ArrayMap = dict[str, np.ndarray]


@dataclass
class MemoryNetworkBenchmarkSpec:
    area_names: list[str]
    ranks: list[int]
    connectome: np.ndarray
    lag: int
    memory: int


@dataclass
class MemoryNetworkSubmission:
    pred_activity: dict[str, np.ndarray]
    pred_messages: np.ndarray | None = None
    pred_latents: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class EvalResults:
    reconstruction: pd.DataFrame
    communication_decode: pd.DataFrame
    message_recovery: pd.DataFrame
    summary: dict[str, float]
    metric_status: dict[str, str]


def load_memory_network_spec(config_dir: str | Path) -> MemoryNetworkBenchmarkSpec:
    config_dir = Path(config_dir)
    model_cfg_path = config_dir / "model" / "model.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Missing model config: {model_cfg_path}")
    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    ranks = [int(v) for v in model_cfg["ranks"]]
    connectome = np.asarray(model_cfg["connectome"], dtype=int)
    return MemoryNetworkBenchmarkSpec(
        area_names=[f"A{i}" for i in range(len(ranks))],
        ranks=ranks,
        connectome=connectome,
        lag=int(model_cfg["lag"]),
        memory=int(model_cfg["memory"]),
    )


def _flatten_time(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _align_lag(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if lag == 0:
        return x, y
    return x[:, lag:, :], y[:, :-lag, :]


def _fit_decode_score(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 0,
) -> float:
    x_flat = _flatten_time(x)
    y_flat = _flatten_time(y).reshape(-1)
    x_train, x_test, y_train, y_test = train_test_split(
        x_flat, y_flat, test_size=test_size, random_state=random_state
    )
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return float(r2_score(y_test, y_pred))


def _fit_poly_decodability_score(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
) -> float:
    reg = PolyRegression(degree)
    reg.ffit(x, y)
    return float(reg.fscore(x, y))


def load_session_arrays(h5_path: str | Path, session: str = "0") -> ArrayMap:
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing file: {h5_path}")
    arrays: ArrayMap = {}
    with h5py.File(h5_path, "r") as h5:
        if session not in h5:
            raise KeyError(f"Session {session!r} missing in {h5_path}")
        g = h5[session]
        for key in g.keys():
            arrays[key] = g[key][:]
    return arrays


def build_submission_from_h5(pred_h5_path: str | Path, area_names: list[str], session: str = "0") -> MemoryNetworkSubmission:
    arr = load_session_arrays(pred_h5_path, session=session)
    pred_activity = {an: arr[f"area-{an}"] for an in area_names if f"area-{an}" in arr}
    meta: dict[str, Any] = {"source": str(pred_h5_path), "session": session}
    if "meta-batch-indices" in arr:
        meta["batch_indices"] = arr["meta-batch-indices"].astype(int)
    if "meta-time-start" in arr:
        mts = np.asarray(arr["meta-time-start"]).reshape(-1)
        meta["time_start"] = int(mts[0]) if len(mts) else 0
    return MemoryNetworkSubmission(
        pred_activity=pred_activity,
        pred_messages=arr.get("message-mesgs"),
        pred_latents=arr.get("message-latents"),
        metadata=meta,
    )


def _align_truth_to_pred(
    truth: np.ndarray,
    pred: np.ndarray,
    batch_indices: np.ndarray | None = None,
    time_start: int | None = None,
) -> np.ndarray:
    if batch_indices is not None:
        truth = truth[batch_indices]
    elif pred.shape[0] != truth.shape[0]:
        raise ValueError(
            f"Batch mismatch without batch_indices metadata: pred={pred.shape[0]}, truth={truth.shape[0]}"
        )

    if truth.shape[1] != pred.shape[1]:
        if time_start is not None and pred.shape[1] == (truth.shape[1] - time_start):
            truth = truth[:, time_start:, ...]
        elif pred.shape[1] < truth.shape[1]:
            # fallback: align by trailing window if exact start offset not provided
            truth = truth[:, truth.shape[1] - pred.shape[1] :, ...]
        else:
            raise ValueError(f"Time mismatch: pred={pred.shape[1]}, truth={truth.shape[1]}")

    return truth


def _build_aligned_truth(
    submission: MemoryNetworkSubmission,
    truth_arrays: ArrayMap,
    spec: MemoryNetworkBenchmarkSpec,
) -> ArrayMap:
    meta = submission.metadata or {}
    batch_indices = meta.get("batch_indices")
    time_start = meta.get("time_start")
    aligned: ArrayMap = {}

    ref_area = spec.area_names[0]
    ref_pred = submission.pred_activity[ref_area]

    for area_name in spec.area_names:
        key = f"area-{area_name}"
        pred = submission.pred_activity[area_name]
        aligned[key] = _align_truth_to_pred(
            truth_arrays[key],
            pred,
            batch_indices=batch_indices,
            time_start=time_start,
        )

    if "truth-inp" in truth_arrays:
        aligned["truth-inp"] = _align_truth_to_pred(
            truth_arrays["truth-inp"],
            np.zeros((ref_pred.shape[0], ref_pred.shape[1], truth_arrays["truth-inp"].shape[-1]), dtype=truth_arrays["truth-inp"].dtype),
            batch_indices=batch_indices,
            time_start=time_start,
        )
    if "message-mesgs" in truth_arrays and submission.pred_messages is not None:
        aligned["message-mesgs"] = _align_truth_to_pred(
            truth_arrays["message-mesgs"],
            submission.pred_messages,
            batch_indices=batch_indices,
            time_start=time_start,
        )
    return aligned


def validate_submission(
    submission: MemoryNetworkSubmission,
    truth_arrays: ArrayMap,
    spec: MemoryNetworkBenchmarkSpec,
) -> dict[str, str]:
    status: dict[str, str] = {}
    aligned_truth = _build_aligned_truth(submission, truth_arrays, spec)
    for area_name in spec.area_names:
        if area_name not in submission.pred_activity:
            raise KeyError(f"Missing required prediction array for area: {area_name}")
        pred = submission.pred_activity[area_name]
        truth = aligned_truth[f"area-{area_name}"]
        if pred.shape != truth.shape:
            raise ValueError(f"Shape mismatch area-{area_name}: pred={pred.shape}, truth={truth.shape}")
    status["reconstruction"] = "ok"
    status["communication_decode"] = "ok"
    status["message_recovery"] = "ok" if submission.pred_messages is not None else "skipped: missing pred_messages"
    return status


def reconstruction_metrics(
    submission: MemoryNetworkSubmission,
    truth_arrays: ArrayMap,
    spec: MemoryNetworkBenchmarkSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for area_name in spec.area_names:
        pred = submission.pred_activity[area_name]
        true = truth_arrays[f"area-{area_name}"]
        mse = float(np.mean((pred - true) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(true.reshape(-1), pred.reshape(-1)))
        corr = _safe_corr(true, pred)
        rows.append({"area": area_name, "mse": mse, "rmse": rmse, "r2": r2, "corr": corr})
    return pd.DataFrame(rows).sort_values("area").reset_index(drop=True)


def communication_recovery_metrics(
    submission: MemoryNetworkSubmission,
    truth_arrays: ArrayMap,
    spec: MemoryNetworkBenchmarkSpec,
    decode_lag: int | None = None,
    decode_degree: int = 1,
) -> pd.DataFrame:
    truth_inp = truth_arrays["truth-inp"]
    lag = spec.lag if decode_lag is None else decode_lag
    rows: list[dict[str, Any]] = []
    for target_idx, target_area in enumerate(spec.area_names):
        x = submission.pred_activity[target_area]
        for src_idx in range(truth_inp.shape[-1]):
            y = truth_inp[:, :, src_idx : src_idx + 1]
            x_aligned, y_aligned = _align_lag(x, y, lag=lag)
            score = _fit_poly_decodability_score(x_aligned, y_aligned, degree=decode_degree)
            rows.append(
                {
                    "target_area": target_area,
                    "target_idx": target_idx,
                    "source_component": src_idx,
                    "decode_r2": score,
                    "decode_lag": lag,
                    "decode_degree": decode_degree,
                    "connectome_truth": int(spec.connectome[target_idx, src_idx]) if src_idx < spec.connectome.shape[1] else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["target_idx", "source_component"]).reset_index(drop=True)


def message_recovery_metrics(
    submission: MemoryNetworkSubmission,
    truth_arrays: ArrayMap,
    spec: MemoryNetworkBenchmarkSpec,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 0,
) -> pd.DataFrame:
    if submission.pred_messages is None:
        return pd.DataFrame(columns=["metric", "value"])
    rows: list[dict[str, Any]] = []
    if "message-mesgs" in truth_arrays and submission.pred_messages.shape == truth_arrays["message-mesgs"].shape:
        pred = submission.pred_messages
        true = truth_arrays["message-mesgs"]
        rows.append(
            {"metric": "message_channel_corr_mean", "value": float(np.nanmean([_safe_corr(true[..., i], pred[..., i]) for i in range(true.shape[-1])]))}
        )
        rows.append({"metric": "message_global_r2", "value": float(r2_score(true.reshape(-1), pred.reshape(-1)))})
    if "truth-inp" in truth_arrays:
        x = submission.pred_messages
        y = truth_arrays["truth-inp"]
        x_aligned, y_aligned = _align_lag(x, y, lag=spec.lag)
        component_scores = []
        for i in range(y_aligned.shape[-1]):
            score = _fit_decode_score(
                x_aligned,
                y_aligned[:, :, i : i + 1],
                alpha=alpha,
                test_size=test_size,
                random_state=random_state,
            )
            component_scores.append(score)
            rows.append({"metric": f"message_to_truth_component_{i}_r2", "value": score})
        rows.append({"metric": "message_to_truth_component_mean_r2", "value": float(np.mean(component_scores))})
    return pd.DataFrame(rows)


def evaluate_memory_network_submission(
    submission: MemoryNetworkSubmission,
    truth_h5_path: str | Path,
    config_dir: str | Path,
    session: str = "0",
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 0,
    decode_lag: int | None = None,
    decode_degree: int = 1,
) -> EvalResults:
    spec = load_memory_network_spec(config_dir)
    truth_arrays = load_session_arrays(truth_h5_path, session=session)
    aligned_truth = _build_aligned_truth(submission, truth_arrays, spec)
    status = validate_submission(submission, truth_arrays, spec)

    recon_df = reconstruction_metrics(submission, aligned_truth, spec)
    comm_df = communication_recovery_metrics(
        submission,
        aligned_truth,
        spec,
        decode_lag=decode_lag,
        decode_degree=decode_degree,
    )
    msg_df = message_recovery_metrics(
        submission,
        aligned_truth,
        spec,
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
    )

    summary = {
        "lag": float(spec.lag),
        "memory": float(spec.memory),
        "recon_r2_mean": float(recon_df["r2"].mean()) if len(recon_df) else np.nan,
        "recon_corr_mean": float(recon_df["corr"].mean()) if len(recon_df) else np.nan,
        "comm_decode_r2_mean": float(comm_df["decode_r2"].mean()) if len(comm_df) else np.nan,
        "comm_decode_r2_connected_mean": float(comm_df.loc[comm_df["connectome_truth"] == 1, "decode_r2"].mean())
        if len(comm_df)
        else np.nan,
        "comm_decode_r2_unconnected_mean": float(comm_df.loc[comm_df["connectome_truth"] == 0, "decode_r2"].mean())
        if len(comm_df)
        else np.nan,
    }
    if len(msg_df):
        for _, row in msg_df.iterrows():
            summary[str(row["metric"])] = float(row["value"])

    return EvalResults(
        reconstruction=recon_df,
        communication_decode=comm_df,
        message_recovery=msg_df,
        summary=summary,
        metric_status=status,
    )


def evaluate_memory_network_h5_submission(
    pred_h5_path: str | Path,
    truth_h5_path: str | Path,
    config_dir: str | Path,
    session: str = "0",
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 0,
    decode_lag: int | None = None,
    decode_degree: int = 1,
) -> EvalResults:
    spec = load_memory_network_spec(config_dir)
    submission = build_submission_from_h5(pred_h5_path, area_names=spec.area_names, session=session)
    return evaluate_memory_network_submission(
        submission=submission,
        truth_h5_path=truth_h5_path,
        config_dir=config_dir,
        session=session,
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        decode_lag=decode_lag,
        decode_degree=decode_degree,
    )
