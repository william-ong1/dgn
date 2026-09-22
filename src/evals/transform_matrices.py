"""Compare DGN vs MR-LFADS input/message transformation matrices.

DGN checkpoints in ``data/`` store the generator RNN maps. MR-LFADS run
checkpoints store the inferred-input (``gen_cell.weight_ih`` ``co`` slice) and
message-emission (``communicator.areas_linear``) maps. These pairs share an
input dimension on the current MN/PD setups (``hidden_size == co_dim`` and
``hidden_size == n_neurons``), so they can be scored with linear CKA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ckpt(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint has no state_dict: {path}")
    return ckpt


def _latest_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "lightning_checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No lightning_checkpoints in {run_dir}")
    for name in ("best.ckpt", "last.ckpt"):
        cand = ckpt_dir / name
        if cand.is_file():
            return cand
    candidates = list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _as_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=np.float64)


def _state_tensor(state: dict, *keys: str) -> np.ndarray:
    for key in keys:
        if key in state:
            return _as_numpy(state[key])
    raise KeyError(f"None of {keys} found in state_dict")


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear CKA between maps that share an input dimension.

    ``A`` is ``(out_a, r)`` and ``B`` is ``(out_b, r)``. Score is in ``[0, 1]``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2-D maps, got {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"CKA requires a shared input dim; got {A.shape[1]} vs {B.shape[1]}"
        )
    if A.size == 0 or B.size == 0:
        return float("nan")

    gram_ab = B @ A.T
    gram_aa = A @ A.T
    gram_bb = B @ B.T
    num = float(np.linalg.norm(gram_ab, ord="fro") ** 2)
    den = float(np.linalg.norm(gram_aa, ord="fro") * np.linalg.norm(gram_bb, ord="fro"))
    if den <= 0.0:
        return float("nan")
    return float(np.clip(num / den, 0.0, 1.0))


def subspace_overlap(A: np.ndarray, B: np.ndarray, k: int | None = None) -> float:
    """Mean squared principal cosines between row-spaces in the shared input dim.

    ``A`` is ``(out_a, r)`` and ``B`` is ``(out_b, r)``. Unlike linear CKA this
    is not driven to 0 just because one map is low-rank. Score is in ``[0, 1]``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2-D maps, got {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Overlap requires a shared input dim; got {A.shape[1]} vs {B.shape[1]}"
        )
    if A.size == 0 or B.size == 0:
        return float("nan")

    _ua, sa, vta = np.linalg.svd(A, full_matrices=False)
    _ub, sb, vtb = np.linalg.svd(B, full_matrices=False)
    ra = int(np.sum(sa > (sa[0] * 1e-6))) if sa.size else 0
    rb = int(np.sum(sb > (sb[0] * 1e-6))) if sb.size else 0
    kk = min(ra, rb) if k is None else min(int(k), ra, rb)
    if kk <= 0:
        return float("nan")
    gram = vta[:kk] @ vtb[:kk].T
    return float(np.clip((np.linalg.norm(gram, ord="fro") ** 2) / kk, 0.0, 1.0))


def discover_dgn_run(experiment_type: str, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    data_dir = root / "data"
    if experiment_type == "memory_network":
        prefix = "memory_network"
    elif experiment_type == "pass_decision":
        prefix = "pass_decision"
    else:
        raise ValueError(f"No DGN run discovery for experiment_type={experiment_type}")

    matches = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_dir()
        and p.name.startswith(prefix)
        and (p / "lightning_checkpoints").is_dir()
    )
    if not matches:
        raise FileNotFoundError(f"No DGN run under {data_dir} starting with {prefix!r}")
    return matches[-1]


def discover_mrlfads_run(experiment_type: str, *, repo_root: Path | None = None) -> Path:
    root = repo_root or REPO_ROOT
    runs_dir = root / "mrlfads_runs"
    if experiment_type == "memory_network":
        prefixes = ("mn_",)
    elif experiment_type == "pass_decision":
        prefixes = ("pd_",)
    else:
        raise ValueError(f"No MR-LFADS run discovery for experiment_type={experiment_type}")

    matches = sorted(
        p
        for p in runs_dir.iterdir()
        if p.is_dir()
        and p.name.startswith(prefixes)
        and (p / "lightning_checkpoints").is_dir()
    )
    if not matches:
        raise FileNotFoundError(
            f"No MR-LFADS run under {runs_dir} starting with {prefixes}"
        )
    return matches[-1]


def _dgn_incoming_slices(
    weight_ih: np.ndarray,
    *,
    area_idx: int,
    connectome: np.ndarray,
    ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split DGN ``weight_ih`` columns into inferred-input vs incoming messages."""
    n_areas = int(connectome.shape[0])
    effectome = ranks.reshape(1, -1) * connectome
    cols_input: list[np.ndarray] = []
    cols_msg: list[np.ndarray] = []
    col = 0
    for src in range(n_areas):
        width = int(effectome[area_idx, src])
        if width <= 0:
            continue
        block = weight_ih[:, col : col + width]
        col += width
        if src == area_idx:
            cols_input.append(block)
        else:
            cols_msg.append(block)
    w_in = np.concatenate(cols_input, axis=1) if cols_input else np.zeros((weight_ih.shape[0], 0))
    w_msg = np.concatenate(cols_msg, axis=1) if cols_msg else np.zeros((weight_ih.shape[0], 0))
    return w_in, w_msg


def extract_dgn_transforms(run_dir: Path, experiment_type: str) -> dict[str, dict[str, np.ndarray]]:
    ckpt = _load_ckpt(_latest_ckpt(run_dir))
    state = ckpt["state_dict"]
    hps = ckpt.get("hyper_parameters") or {}

    out: dict[str, dict[str, np.ndarray]] = {}
    if experiment_type == "memory_network":
        connectome = np.asarray(hps["connectome"], dtype=np.int64)
        ranks = np.asarray(hps["ranks"], dtype=np.int64).reshape(-1)
        area_names = [f"A{i}" for i in range(len(ranks))]
        for ia, name in enumerate(area_names):
            w_ih = _state_tensor(state, f"areas.{name}.rnn.model.weight_ih")
            w_out = _state_tensor(state, f"areas.{name}.output.model.linear0.weight")
            w_in, w_msg_in = _dgn_incoming_slices(
                w_ih, area_idx=ia, connectome=connectome, ranks=ranks
            )
            rank = int(ranks[ia])
            out[name] = {
                "inferred_input": w_in.T.copy(),  # (k_in, hidden)
                "message_in": w_msg_in.T.copy(),
                "message_out": w_out[:rank].copy(),  # (rank, hidden)
            }
        return out

    if experiment_type == "pass_decision":
        w_p = _state_tensor(state, "P_area.rnn.model.weight_ih")
        w_d = _state_tensor(state, "D_area.rnn.model.weight_ih")
        w_p_out = _state_tensor(state, "P_area.output.model.linear0.weight")
        out["P"] = {
            "inferred_input": w_p.T.copy(),
            "message_in": np.zeros((0, w_p.shape[0])),
            "message_out": w_p_out.copy(),
        }
        out["D"] = {
            "inferred_input": np.zeros((0, w_d.shape[0])),
            "message_in": w_d.T.copy(),
            "message_out": np.zeros((0, w_d.shape[0])),
        }
        return out

    raise ValueError(f"Unsupported experiment_type: {experiment_type}")


def _mrl_area_dims(hps: dict, area_name: str) -> tuple[int, int, int, int]:
    from omegaconf import OmegaConf

    raw = hps["areas_params"][area_name]
    area = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else dict(raw)
    gen_dim = int(area["gen_dim"])
    co_dim = int(area["co_dim"])
    com_dim = int(area["com_dim"])
    neurons = area.get("num_neurons") or {}
    if isinstance(neurons, dict):
        n_neurons = int(sum(int(v) for v in neurons.values()))
    else:
        n_neurons = int(neurons)
    return gen_dim, co_dim, com_dim, n_neurons


def extract_mrlfads_transforms(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    ckpt = _load_ckpt(_latest_ckpt(run_dir))
    state = ckpt["state_dict"]
    hps = ckpt.get("hyper_parameters") or {}
    num_other = int(hps.get("num_other_areas", 0))

    area_names = list(hps["areas_params"].keys())
    out: dict[str, dict[str, np.ndarray]] = {}
    emit_from: dict[str, list[np.ndarray]] = {name: [] for name in area_names}

    for name in area_names:
        _gen_dim, co_dim, com_dim, n_neurons = _mrl_area_dims(hps, name)
        w_ih = _state_tensor(state, f"areas.{name}.decoder.gen_cell.weight_ih")
        com_total = com_dim * num_other
        expected_in = co_dim + com_total
        if w_ih.shape[1] < expected_in:
            raise ValueError(
                f"{name} gen_cell.weight_ih has {w_ih.shape[1]} cols, expected >= {expected_in}"
            )
        w_co = w_ih[:, :co_dim]
        w_com = w_ih[:, co_dim : co_dim + com_total]

        for src in area_names:
            if src == name:
                continue
            key = f"areas.{name}.communicator.areas_linear.{src}.weight"
            if key not in state:
                continue
            w_emit = _as_numpy(state[key])
            emit_from[src].append(w_emit[:com_dim])

        out[name] = {
            "inferred_input": w_co.copy(),
            "message_in": w_com.copy(),
        }

    for src, blocks in emit_from.items():
        out[src]["message_out"] = (
            np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 0))
        )
    return out


def extract_dgn_incoming_weights(
    run_dir: Path, experiment_type: str
) -> dict[tuple[str, str], np.ndarray]:
    """Target-region input-weight columns for each (target, source) edge.

    Values are ``weight_ih`` slices of shape ``(hidden_or_gates, n_channels)``.
    """
    ckpt = _load_ckpt(_latest_ckpt(run_dir))
    state = ckpt["state_dict"]
    hps = ckpt.get("hyper_parameters") or {}
    blocks: dict[tuple[str, str], np.ndarray] = {}

    if experiment_type == "memory_network":
        connectome = np.asarray(hps["connectome"], dtype=np.int64)
        ranks = np.asarray(hps["ranks"], dtype=np.int64).reshape(-1)
        effectome = ranks.reshape(1, -1) * connectome
        names = [f"A{i}" for i in range(len(ranks))]
        for ia, tgt in enumerate(names):
            w_ih = _state_tensor(state, f"areas.{tgt}.rnn.model.weight_ih")
            col = 0
            for js, src in enumerate(names):
                width = int(effectome[ia, js])
                if width <= 0:
                    continue
                blocks[(tgt, src)] = w_ih[:, col : col + width].copy()
                col += width
        return blocks

    if experiment_type == "pass_decision":
        w_p = _state_tensor(state, "P_area.rnn.model.weight_ih")
        w_d = _state_tensor(state, "D_area.rnn.model.weight_ih")
        blocks[("P", "P")] = w_p.copy()
        blocks[("D", "P")] = w_d.copy()
        return blocks

    if experiment_type == "multi_task":
        from .eval_utils import load_multi_task_spec, mt_external_input_dim, mt_predecessors

        spec = None
        if hps.get("diagram") and hps.get("num_areas") is not None:
            try:
                from omegaconf import OmegaConf

                diagram = hps["diagram"]
                if OmegaConf.is_config(diagram):
                    diagram = OmegaConf.to_container(diagram, resolve=True)
                stim = hps.get("stim_input_areas") or []
                if OmegaConf.is_config(stim):
                    stim = OmegaConf.to_container(stim, resolve=True)
                spec = {
                    "num_areas": int(hps["num_areas"]),
                    "num_channels": int(hps.get("num_channels", 8)),
                    "num_angles": int(hps.get("num_angles", 36)),
                    "delay": int(hps.get("delay", 0)),
                    "diagram": [
                        (str(e[0]), str(e[1]), str(e[2]) if len(e) > 2 else "1")
                        for e in (diagram or [])
                        if len(e) >= 2
                    ],
                    "stim_input_areas": list(stim),
                    "area_names": [f"A{i}" for i in range(int(hps["num_areas"]))],
                }
            except Exception:
                spec = None
        if spec is None:
            spec = load_multi_task_spec(run_dir / "configs")

        for tgt in spec["area_names"]:
            w_ih = _state_tensor(state, f"areas.{tgt}.rnn.model.weight_ih")
            col = mt_external_input_dim(spec, tgt)
            for src in mt_predecessors(spec, tgt):
                width = int(spec["num_channels"])
                if col + width > w_ih.shape[1]:
                    break
                blocks[(tgt, src)] = w_ih[:, col : col + width].copy()
                col += width
        return blocks

    raise ValueError(f"Unsupported experiment_type: {experiment_type}")


def extract_mrl_incoming_weights(
    run_dir: Path,
) -> tuple[dict[tuple[str, str], np.ndarray], list[str], dict[str, int]]:
    """MR-LFADS ``gen_cell.weight_ih`` communication columns per (target, source)."""
    ckpt = _load_ckpt(_latest_ckpt(run_dir))
    state = ckpt["state_dict"]
    hps = ckpt.get("hyper_parameters") or {}
    area_names = list(hps["areas_params"].keys())
    blocks: dict[tuple[str, str], np.ndarray] = {}
    com_dims: dict[str, int] = {}

    for tgt in area_names:
        _gen_dim, co_dim, com_dim, _n = _mrl_area_dims(hps, tgt)
        com_dims[tgt] = com_dim
        w_ih = _state_tensor(state, f"areas.{tgt}.decoder.gen_cell.weight_ih")
        others = [n for n in area_names if n != tgt]
        for ioa, src in enumerate(others):
            start = co_dim + ioa * com_dim
            blocks[(tgt, src)] = w_ih[:, start : start + com_dim].copy()
    return blocks, area_names, com_dims


def compare_transform_matrices(
    experiment_type: str,
    *,
    dgn_run_dir: Path | None = None,
    mrlfads_run_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Return per-area and mean linear-CKA scores.

    ``inferred-input-transform-overlap``
        Subspace overlap between the DGN incoming input map (transposed) and
        the MR-LFADS ``gen_cell`` inferred-input (``co``) columns.
    ``message-transform-overlap``
        Subspace overlap between the DGN outgoing message readout and the
        MR-LFADS communicator emission (``areas_linear`` mean head).
    Linear CKA is also reported; it is typically near 0 when one map is rank-1.
    """
    root = repo_root or REPO_ROOT
    dgn_dir = Path(dgn_run_dir) if dgn_run_dir is not None else discover_dgn_run(
        experiment_type, repo_root=root
    )
    mrl_dir = (
        Path(mrlfads_run_dir)
        if mrlfads_run_dir is not None
        else discover_mrlfads_run(experiment_type, repo_root=root)
    )

    dgn = extract_dgn_transforms(dgn_dir, experiment_type)
    mrl = extract_mrlfads_transforms(mrl_dir)

    per_area: dict[str, dict[str, float]] = {}
    input_overlap: list[float] = []
    message_overlap: list[float] = []
    input_cka: list[float] = []
    message_cka: list[float] = []

    for area, dgn_maps in dgn.items():
        row: dict[str, float] = {}
        if area in mrl and dgn_maps["inferred_input"].size and mrl[area]["inferred_input"].size:
            w_dgn = dgn_maps["inferred_input"]
            w_mrl = mrl[area]["inferred_input"]
            if w_dgn.shape[1] == w_mrl.shape[1]:
                overlap = subspace_overlap(w_dgn, w_mrl)
                cka = linear_cka(w_dgn, w_mrl)
                row["inferred-input-transform-overlap"] = overlap
                row["inferred-input-transform-cka"] = cka
                input_overlap.append(overlap)
                input_cka.append(cka)

        w_dgn_out = dgn_maps["message_out"]
        w_mrl_out = mrl.get(area, {}).get("message_out")
        if (
            w_dgn_out.size
            and w_mrl_out is not None
            and w_mrl_out.size
            and w_dgn_out.shape[1] == w_mrl_out.shape[1]
        ):
            overlap = subspace_overlap(w_dgn_out, w_mrl_out)
            cka = linear_cka(w_dgn_out, w_mrl_out)
            row["message-transform-overlap"] = overlap
            row["message-transform-cka"] = cka
            message_overlap.append(overlap)
            message_cka.append(cka)

        if row:
            per_area[area] = row

    results: dict[str, Any] = {"by-area": per_area}
    if input_overlap:
        results["inferred-input-transform-overlap"] = float(np.mean(input_overlap))
        results["inferred-input-transform-cka"] = float(np.mean(input_cka))
    if message_overlap:
        results["message-transform-overlap"] = float(np.mean(message_overlap))
        results["message-transform-cka"] = float(np.mean(message_cka))
    results["dgn-run"] = str(dgn_dir)
    results["mrlfads-run"] = str(mrl_dir)
    return results


if __name__ == "__main__":
    import json

    for exp in ("memory_network", "pass_decision"):
        dgn_dir = discover_dgn_run(exp)
        mrl_dir = discover_mrlfads_run(exp)
        dgn = extract_dgn_transforms(dgn_dir, exp)
        mrl = extract_mrlfads_transforms(mrl_dir)
        print(f"\n=== {exp} shapes ===")
        print("dgn", dgn_dir.name)
        for area, maps in dgn.items():
            print(
                f"  {area}: in={maps['inferred_input'].shape} "
                f"msg_in={maps['message_in'].shape} "
                f"msg_out={maps['message_out'].shape}"
            )
        print("mrl", mrl_dir.name)
        for area, maps in mrl.items():
            print(
                f"  {area}: in={maps['inferred_input'].shape} "
                f"msg_in={maps['message_in'].shape} "
                f"msg_out={maps.get('message_out', np.zeros((0, 0))).shape}"
            )
        out = compare_transform_matrices(exp, dgn_run_dir=dgn_dir, mrlfads_run_dir=mrl_dir)
        print(json.dumps({k: v for k, v in out.items() if k != "by-area"}, indent=2, default=str))
        print("by-area:", json.dumps(out.get("by-area", {}), indent=2))
