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


# Slice the truth arrays along time dimension so truth arrays align with the submission time 0.
def slice_truth_for_time_alignment(truth: ArrayMap, *, truth_time_start: int, pred_time_len: int) -> ArrayMap:
    if truth_time_start < 0:
        raise ValueError(f"truth_time_start must be >= 0, got {truth_time_start}")

    end = truth_time_start + pred_time_len
    out: ArrayMap = {}
    must_align_prefixes = ("area-", "message-")

    for key, arr in truth.items():
        if key.startswith("meta-"):
            out[key] = arr
            continue
        if not isinstance(arr, np.ndarray) or arr.ndim < 2:
            out[key] = arr
            continue

        must_align = key.startswith(must_align_prefixes)
        if arr.shape[1] < end:
            if must_align:
                raise ValueError(
                    f"Truth dataset {key!r} has time dim {arr.shape[1]} < {end} "
                    f"(truth_time_start={truth_time_start} + pred_time_len={pred_time_len})."
                )
            # Trial-level auxiliaries (e.g. truth-amp1 with T=1) — leave unchanged.
            out[key] = arr
            continue

        out[key] = arr[:, truth_time_start:end]

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
    x = np.asarray(activity, dtype=np.float64)
    sd = float(np.std(x))
    if sd >= 1e-12:
        x = (x - float(np.mean(x))) / sd
    rates = rate_max * (x + 1.0) / 2.0
    lam = np.clip(rates * dt, a_min=0.0, a_max=rate_max)
    return lam.astype(np.float32)


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


def resolve_dataset_config_dir(truth_h5: Path | str, fallback: Path | str) -> Path:
    """
    Prefer the DGN config saved alongside ``data.h5`` (per-dataset diagram / graph).

    Falls back to the evaluation config directory when the dataset has no bundled
    ``configs/model/model.yaml`` (e.g. generic ``configs/multi_task/``).
    """
    truth_path = Path(truth_h5).expanduser().resolve()
    bundled = truth_path.parent / "configs"
    if (bundled / "model" / "model.yaml").is_file():
        return bundled.resolve()
    return Path(fallback).expanduser().resolve()


def load_multi_task_effectome(config_dir: Path) -> tuple[np.ndarray, list[str]]:
    """
    Build a ground-truth effectome matrix from a MultiTaskNet ``diagram``.

    Returns ``effectome[target, source]`` with edge weight ``num_channels`` for
    each inter-area edge among recurrent areas ``A0 … A{num_areas-1}``.
    """
    model_cfg = config_dir / "model" / "model.yaml"
    if not model_cfg.is_file():
        raise FileNotFoundError(f"Multi-task model config not found: {model_cfg}")

    with model_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_areas = int(cfg["num_areas"])
    num_channels = float(cfg.get("num_channels", 1))
    area_names = [f"A{i}" for i in range(num_areas)]
    name_to_idx = {name: idx for idx, name in enumerate(area_names)}

    effectome = np.zeros((num_areas, num_areas), dtype=np.float64)
    for edge in cfg.get("diagram") or []:
        if len(edge) < 2:
            continue
        src, dst = str(edge[0]), str(edge[1])
        if src not in name_to_idx or dst not in name_to_idx:
            continue
        effectome[name_to_idx[dst], name_to_idx[src]] = num_channels

    return effectome, area_names


def load_mrlfads_ic_enc_seq_len(run_dir: Path) -> int:
    """Read ``ic_enc_seq_len`` from an MR-LFADS run's saved model config."""
    model_cfg = run_dir / "configs" / "model" / "model.yaml"
    if not model_cfg.is_file():
        raise FileNotFoundError(f"MR-LFADS model config not found: {model_cfg}")

    with model_cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return int(cfg["ic_enc_seq_len"])


def resolve_mrlfads_run_data_h5(run_dir: Path) -> Path:
    """
    Resolve the training ``data.h5`` path from a run's saved datamodule config.

    ``BasicDataModule`` loads ``{datapath_override}/{filename}/data.h5`` (``filename``
    may be empty).
    """
    dm_cfg_path = run_dir / "configs" / "datamodule" / "datamodule.yaml"
    if not dm_cfg_path.is_file():
        raise FileNotFoundError(f"MR-LFADS datamodule config not found: {dm_cfg_path}")

    with dm_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    datapath = cfg.get("datapath_override")
    if not datapath:
        raise ValueError(
            f"{dm_cfg_path} has no datapath_override; pass --input-h5 explicitly."
        )

    filename = str(cfg.get("filename") or "")
    data_h5 = Path(datapath) / filename / "data.h5"
    if not data_h5.is_file():
        raise FileNotFoundError(
            f"Training data.h5 not found for run {run_dir.name}: {data_h5}"
        )
    return data_h5.resolve()


def is_mrlfads_run_dir(path: Path) -> bool:
    """True when ``path`` looks like a trained MR-LFADS run directory."""
    return (
        path.is_dir()
        and (path / "configs" / "main.yaml").is_file()
        and (path / "lightning_checkpoints").is_dir()
        and any((path / "lightning_checkpoints").glob("*.ckpt"))
    )


def discover_mrlfads_runs(
    runs_dir: Path,
    *,
    run_glob: str | None = None,
    recursive: bool = False,
) -> list[Path]:
    """
    Find MR-LFADS run directories under ``runs_dir``.

    When ``recursive`` is True, also searches nested subdirectories (e.g.
    ``rt_go/rt_go_kl0001_id…/``).
    """
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    if recursive:
        candidates = [
            path
            for path in sorted(runs_dir.rglob("*"))
            if path.is_dir() and is_mrlfads_run_dir(path)
        ]
    else:
        candidates = []
        for path in sorted(runs_dir.iterdir()):
            if not path.is_dir():
                continue
            if not (path / "configs" / "main.yaml").is_file():
                continue
            ckpt_dir = path / "lightning_checkpoints"
            if not ckpt_dir.is_dir() or not any(ckpt_dir.glob("*.ckpt")):
                print(f"skip {path.name}: no checkpoint")
                continue
            candidates.append(path)

    if run_glob:
        candidates = [p for p in candidates if p.match(run_glob)]

    return candidates


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


def _strip_aggregate_mean_suffix(name: str) -> str:
    """Map aggregate keys like ``r2-held-out-mean`` -> ``r2-held-out`` for row alignment."""
    return name[: -len("-mean")] if name.endswith("-mean") else name


def results_to_summary_rows(run_name: str, results: dict) -> list[dict]:
    """
    Flatten eval results into summary rows.

    - One row per area/region (``region=P``, ``region=D``, ...)
    - When there are 2+ areas, also one ``region=combined`` row with aggregate
      means plus structure metrics (message/effectome).
    - Single-area runs: just the one region row (no redundant combined).
    """
    region_df, global_df, temporal_df = split_results_for_display(results)
    rows: list[dict] = []

    if not region_df.empty:
        for _, r in region_df.iterrows():
            row: dict = {"run": run_name, "region": r["region"]}
            for col in region_df.columns:
                if col == "region":
                    continue
                row[col] = r[col]
            rows.append(row)

    n_regions = len(rows)
    if n_regions > 1 or (n_regions == 0 and not global_df.empty):
        crow: dict = {"run": run_name, "region": "combined"}
        if not global_df.empty:
            for col in global_df.columns:
                crow[_strip_aggregate_mean_suffix(col)] = global_df.iloc[0][col]
        if not temporal_df.empty:
            for col in temporal_df.columns:
                crow[f"temporal-{col}"] = temporal_df.iloc[0][col]
        rows.append(crow)
    elif n_regions == 1 and not temporal_df.empty:
        # Attach temporal metrics onto the single-area row when present
        for col in temporal_df.columns:
            rows[0][f"temporal-{col}"] = temporal_df.iloc[0][col]

    return rows
