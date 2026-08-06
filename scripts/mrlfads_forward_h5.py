#!/usr/bin/env python3
"""
Load an MR-LFADS run, run validation (forward pass) on an HDF5 file of observed area activity,
and write predicted area activity plus neuron-index metadata to an output HDF5.

For each area, **held-out** neurons (``hn_indices``) are filled from ``model.preds`` (predictor);
other neurons use ``model.outputs``.

- **Gaussian** (and similar): predictive **mean** via ``output_dist.unbind`` + ``compute_means``.
- **Poisson**: readout stores **log-rate**; exported values are ``exp(log_rate)`` (firing rates), matching MR-LFADS plots.

Use ``--output-dist gaussian`` or ``--output-dist poisson`` to match each area's ``output_dist`` (readout / observation model).

Use ``--experiment-type pass_decision`` for pass-decision runs (areas P/D): exports
``message-p_to_d`` instead of memory-network message tensors, and uses a fixed P→D
effectome target at evaluation time (no connectome config). Downstream eval
(``batch_mrlfads_eval`` / ``truth_input_decoding_pass_decision``) decodes
``cumsum`` from that predicted P→D message (expected low R²).

Single-area runs (``num_other_areas=0``) export only neural activity, held-out indices,
and optional ``region-factors`` — no effectome or message tensors.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pytorch_lightning as pl
import torch


# Helper function to reorder predictions to input trials
def reorder_predictions_to_input_trials(preds: np.ndarray, original_trial_idx: np.ndarray) -> np.ndarray:
    out = np.empty_like(preds)
    out[original_trial_idx] = preds
    return out

# Helper function to get held-out neuron indices
def _held_out_neuron_indices(model: Any, area_name: str, sess: int = 0) -> np.ndarray:
    hn = getattr(model.hparams, "hn_indices", None)
    if hn is None:
        return np.array([], dtype=np.int64)
    raw = hn[area_name]
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    if isinstance(raw, list) and len(raw) and isinstance(raw[0], (list, tuple)):
        flat = list(raw[sess])
    else:
        flat = list(raw)
    return np.asarray(flat, dtype=np.int64)


OutputDist = Literal["gaussian", "poisson"]
ExperimentType = Literal["memory_network", "pass_decision"]


def _readout_to_rate_numpy(area: Any, raw: torch.Tensor, output_dist: OutputDist) -> np.ndarray:
    """Match MR-LFADS callbacks: Poisson → exp(log-rate); else Gaussian-style means."""
    raw_d = raw.detach().cpu()
    if output_dist == "poisson":
        return torch.exp(raw_d).numpy().astype(np.float32)
    out_params = area.output_dist.unbind(raw_d)
    return area.output_dist.compute_means(out_params).numpy().astype(np.float32)


def _merged_area_predictive_means(
    model: Any,
    area_name: str,
    sess: int,
    *,
    output_dist: OutputDist,
) -> np.ndarray:
    """
    Rates / predictive means from ``outputs``, with held-out columns from ``preds``.
    """
    area = model.areas[area_name]
    raw_out = model.outputs[area_name][sess]
    pred_mean = _readout_to_rate_numpy(area, raw_out, output_dist)

    hn_idx = _held_out_neuron_indices(model, area_name, sess)
    preds_dict = getattr(model, "preds", None)
    if (
        hn_idx.size == 0
        or preds_dict is None
        or area_name not in preds_dict
        or sess >= len(preds_dict[area_name])
    ):
        return pred_mean

    raw_pr = preds_dict[area_name][sess]
    if raw_pr.shape[-1] == 0:
        return pred_mean

    pred_mean_ho = _readout_to_rate_numpy(area, raw_pr, output_dist)

    if pred_mean_ho.shape[-1] != hn_idx.size:
        raise ValueError(
            f"{area_name} sess={sess}: preds width {pred_mean_ho.shape[-1]} "
            f"!= len(hn_indices)={hn_idx.size}"
        )

    merged = np.copy(pred_mean)
    merged[..., hn_idx] = pred_mean_ho
    return merged


def _region_factors_from_model(model: Any) -> np.ndarray:
    latent_slices: list[np.ndarray] = []
    for area_name in model.area_names:
        ahps = model.areas[area_name].hparams
        states = model.save_var[area_name].states.detach().cpu().numpy()
        latent_slices.append(states[:, 1:, -int(ahps.fac_dim) :])
    return np.concatenate(latent_slices, axis=-1).astype(np.float32)


def _extract_messages_memory_network(model: Any) -> np.ndarray:
    """
    Extract memory-network message predictions from MR-LFADS internals.

    ``message-mesgs``: concatenation across areas of communication posterior means
    from ``save_var[area].com_params[:, :, :msg_dim]``.
    """
    comm_slices: list[np.ndarray] = []
    for area_name in model.area_names:
        ahps = model.areas[area_name].hparams
        msg_dim = int(ahps.com_dim) * int(model.hparams.num_other_areas)
        com_params = model.save_var[area_name].com_params.detach().cpu().numpy()
        comm_slices.append(com_params[:, :, :msg_dim])
    return np.concatenate(comm_slices, axis=-1).astype(np.float32)


def _extract_message_p_to_d(model: Any) -> np.ndarray:
    """
    Pass-decision export: predicted pass→decision channel from decision-area com_params.
    """
    decision_area = next((n for n in model.area_names if str(n).lower().startswith("d")), None)
    if decision_area is None:
        raise ValueError(
            "pass_decision export requires a decision area (name starting with 'D'); "
            f"got areas={list(model.area_names)}"
        )

    ahps = model.areas[decision_area].hparams
    msg_dim = int(ahps.com_dim) * int(model.hparams.num_other_areas)
    com_params = model.save_var[decision_area].com_params.detach().cpu().numpy()
    return com_params[:, :, :msg_dim].astype(np.float32)


def _has_cross_area_communication(model: Any) -> bool:
    """True when the model has inter-area communication channels (num_other_areas > 0)."""
    return int(getattr(model.hparams, "num_other_areas", 0)) > 0


def _compute_effectome_from_model(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Match the provided `volume(model, reduction=[-1, -2, -3])` logic.

    Returns:
        comm_scores: continuous communication volume, shape
            (num_target_areas, num_source_areas)
        inferred_input_scores: continuous inferred-input volume, shape
            (num_areas,)
    """
    hps = model.hparams
    num_areas = hps.num_other_areas + 1

    batch, time = model.save_var[model.area_names[0]].inputs.shape[:2]

    volume = [
        np.zeros(
            (
                num_areas,
                batch,
                time,
                model.areas[model.area_names[i]].hparams.com_dim,
            )
        )
        for i in range(num_areas)
    ]

    uvolume = [
        np.zeros(
            (
                batch,
                time,
                model.areas[model.area_names[i]].hparams.co_dim,
            )
        )
        for i in range(num_areas)
    ]

    for ia, (area_name, area) in enumerate(model.areas.items()):
        ahps = area.hparams

        inputs = model.save_var[area_name].inputs.detach().cpu()
        ci_enc_dim = ahps.ci_enc_dim
        com_dim = ahps.com_dim
        co_dim = ahps.co_dim

        _, com, co = torch.split(
            inputs,
            [ci_enc_dim, com_dim * hps.num_other_areas, co_dim],
            dim=2,
        )

        for ioa in range(hps.num_other_areas):
            idx = ioa + 1 if ioa >= ia else ioa
            volume[ia][idx] = (
                com[..., com_dim * ioa : com_dim * (ioa + 1)]
                .numpy()
            )

        uvolume[ia] = co.numpy()

    axes = (-1, -2, -3)

    comm_scores = [
        np.sqrt(np.sum(np.square(volume[ia]), axis=axes))
        for ia in range(num_areas)
    ]

    inferred_input_scores = [
        np.sqrt(np.sum(np.square(uvolume[ia]), axis=axes))
        for ia in range(num_areas)
    ]

    return np.array(comm_scores), np.array(inferred_input_scores)


# Main function to write the predictions to an HDF5 file (single session ``0`` only)
def write_mrlfads_area_activity_h5(
    *,
    model: Any,
    datamodule: Any,
    out_path: str | Path,
    output_dist: OutputDist,
    experiment_type: ExperimentType = "memory_network",
) -> Path:
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()

    trial_idx = np.asarray(datamodule.val_session_indices[0], dtype=np.int64)
    with h5py.File(out_path, "w") as dst:
        g = dst.create_group("0")

        # Write each area's predictions (held-out channels from ``preds``, rest from ``outputs``)
        for area_name in model.area_names:
            pred_mean = _merged_area_predictive_means(
                model, area_name, sess=0, output_dist=output_dist
            )
            aligned = reorder_predictions_to_input_trials(pred_mean, trial_idx)
            ds = g.create_dataset(f"area-{area_name}", data=aligned)
            ds.attrs["type"] = "hidden_state"
            if output_dist == "poisson":
                ds.attrs["representation"] = "mrlfads_poisson_rate"
            else:
                ds.attrs["representation"] = "mrlfads_gaussian_mean"

        # Held-out neuron indices (MR-LFADS ``hn_indices``) — indices along the neuron dim of ``area-*``.
        for area_name in model.area_names:
            ho = _held_out_neuron_indices(model, area_name, sess=0)
            ds = g.create_dataset(f"meta-held-out-neuron-indices-area-{area_name}", data=ho)
            ds.attrs["role"] = "held_out_train_time"
            ds.attrs["description"] = "Subset of neuron indices held out during MR-LFADS training (hn_indices)."


        has_comm = _has_cross_area_communication(model)

        if has_comm:
            comm_scores, inferred_input_scores = _compute_effectome_from_model(model)

            ds = g.create_dataset("effectome-scores", data=comm_scores.astype(np.float32))
            ds.attrs["type"] = "communication_volume"
            ds.attrs["description"] = (
                "Continuous communication volume matching volume(model, reduction=[-1, -2, -3])."
            )

            ds = g.create_dataset("inferred-input-scores", data=inferred_input_scores.astype(np.float32))
            ds.attrs["type"] = "inferred_input_volume"
            ds.attrs["description"] = (
                "Continuous inferred-input volume matching volume(model, reduction=[-1, -2, -3])."
            )

        region_factors = reorder_predictions_to_input_trials(
            _region_factors_from_model(model), trial_idx
        )

        if experiment_type == "memory_network":
            if has_comm:
                message_mesgs = reorder_predictions_to_input_trials(
                    _extract_messages_memory_network(model), trial_idx
                )

                ds = g.create_dataset("message-mesgs", data=message_mesgs)
                ds.attrs["type"] = "prediction"
                ds.attrs["description"] = (
                    "Concatenated communication posterior means from save_var[area].com_params "
                    "for each area."
                )

                ds = g.create_dataset("message-latents", data=region_factors)
                ds.attrs["type"] = "prediction"
                ds.attrs["description"] = (
                    "Concatenated factor states from save_var[area].states[:, 1:, -fac_dim:] "
                    "for each area."
                )

            ds = g.create_dataset("region-factors", data=region_factors)
            ds.attrs["type"] = "prediction"
            ds.attrs["description"] = (
                "Concatenated factor states from save_var[area].states[:, 1:, -fac_dim:] for each area."
            )
        elif experiment_type == "pass_decision":
            if has_comm:
                message_p_to_d = reorder_predictions_to_input_trials(
                    _extract_message_p_to_d(model), trial_idx
                )

                ds = g.create_dataset("message-p_to_d", data=message_p_to_d)
                ds.attrs["type"] = "prediction"
                ds.attrs["description"] = (
                    "Predicted pass-to-decision communication channel from decision-area com_params."
                )

            ds = g.create_dataset("region-factors", data=region_factors)
            ds.attrs["type"] = "prediction"
            ds.attrs["description"] = (
                "Concatenated factor states from save_var[area].states[:, 1:, -fac_dim:] for each area."
            )
        else:
            raise ValueError(f"unsupported experiment_type: {experiment_type}")



    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="MR-LFADS run directory (contains configs/main.yaml and lightning_checkpoints/).",
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="Ground truth HDF5 with session groups and observed neural activity in area-A0, area-A1, ...",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        required=True,
        help="Path to write predictions (created or overwritten): area-* plus meta neuron index datasets.",
    )
    parser.add_argument(
        "--accelerator",
        default="cpu",
        help="PyTorch Lightning accelerator for validate() (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--output-dist",
        type=str,
        required=True,
        choices=("gaussian", "poisson"),
        help="Must match area output_dist: poisson → exp(log-rate); gaussian → compute_means.",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="memory_network",
        choices=("memory_network", "pass_decision"),
        help=(
            "memory_network: export message-mesgs/message-latents; "
            "pass_decision: export message-p_to_d (fixed P→D effectome at eval)."
        ),
    )
    args = parser.parse_args()

    # scripts/ -> repo root
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "mrlfads2"))

    import mrlfads.datamodules as mdm
    import mrlfads.paths as mpaths
    from mrlfads.run import run

    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "configs" / "main.yaml"
    input_h5 = args.input_h5.expanduser().resolve()
    output_h5 = args.output_h5.expanduser().resolve()

    if not config_path.is_file():
        raise SystemExit(f"Missing config: {config_path}")
    if not input_h5.is_file():
        raise SystemExit(f"Missing input H5: {input_h5}")

    mpaths.datapath = str(repo_root / "datasets")
    mdm.path.datapath = mpaths.datapath

    model, datamodule, _ckpt = run(
        config_path=str(config_path),
        train=False,
        checkpoint_dir=str(run_dir),
    )

    datamodule.hparams.p_split = [0.0, 1.0]
    datamodule.setup()

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.validate(model, datamodule=datamodule, verbose=False)

    write_mrlfads_area_activity_h5(
        model=model,
        datamodule=datamodule,
        out_path=output_h5,
        output_dist=args.output_dist,
        experiment_type=args.experiment_type,
    )
    print(f"Wrote {output_h5}")


if __name__ == "__main__":
    main()
