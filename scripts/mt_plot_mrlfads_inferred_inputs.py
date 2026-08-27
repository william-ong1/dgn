#!/usr/bin/env python3
"""
Plot MR-LFADS inferred inputs (controller outputs ``co``) for a trained run.

After a validation forward pass, extracts per-area inferred inputs from
``save_var[area].inputs`` (last ``co_dim`` channels) and writes:

  1. Activity bars — RMS over batch×time per channel (which dims are active)
  2. Mean time courses — trial-averaged ``co`` vs time, one line per channel

Example (multi-task):

    python scripts/mt_plot_mrlfads_inferred_inputs.py \\
        --run-dir /path/to/dly_go_kl01_id... \\
        --input-h5 /path/to/data.h5 \\
        --output-dir /path/to/out \\
        --accelerator cpu

If ``--input-h5`` is omitted, tries the training ``data.h5`` from the run's
datamodule config (``datapath_override``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch


def _setup_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "mrlfads2"))
    sibling = repo_root.parent / "Projects" / "mrlfads2"
    if sibling.is_dir():
        sys.path.insert(0, str(sibling))


def _patch_torch_load() -> None:
    _torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _torch_load_compat


def extract_inferred_inputs(model) -> dict[str, np.ndarray]:
    """
    Return ``{area_name: co}`` with ``co`` shape ``(batch, time, co_dim)``.

    Splits ``save_var[area].inputs`` into ``[ci | messages | inferred_inputs]``.
    """
    hps = model.hparams
    out: dict[str, np.ndarray] = {}
    for area_name, area in model.areas.items():
        ahps = area.hparams
        if not getattr(ahps, "use_con", False) or ahps.co_dim <= 0:
            print(f"  skip {area_name}: no controller / co_dim=0", flush=True)
            continue
        inputs = model.save_var[area_name].inputs.detach().cpu()
        _, _, co = torch.split(
            inputs,
            [
                ahps.ci_enc_dim,
                ahps.com_dim * hps.num_other_areas,
                ahps.co_dim,
            ],
            dim=2,
        )
        out[area_name] = co.numpy().astype(np.float32)
        print(
            f"  {area_name}: co shape={out[area_name].shape}  "
            f"(co_dim={ahps.co_dim})",
            flush=True,
        )
    return out


def channel_activity(co: np.ndarray) -> np.ndarray:
    """RMS over batch and time → (co_dim,)."""
    return np.sqrt(np.mean(np.square(co), axis=(0, 1)))


def plot_inferred_inputs(
    co_by_area: dict[str, np.ndarray],
    output_dir: Path,
    *,
    run_name: str,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Write activity-bar and time-course PNGs. Returns their paths."""
    areas = list(co_by_area.keys())
    n = len(areas)
    if n == 0:
        raise SystemExit("No areas with inferred inputs to plot.")

    # ----- activity bars -----
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), squeeze=False, sharey=True)
    for ax, area in zip(axes[0], areas):
        act = channel_activity(co_by_area[area])
        x = np.arange(len(act))
        ax.bar(x, act, color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x], fontsize=8)
        ax.set_xlabel("co channel")
        ax.set_title(area)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("RMS activity")
    fig.suptitle(f"{run_name}\ninferred-input channel activity", fontsize=11)
    fig.tight_layout()
    bars_path = output_dir / f"{run_name}_inferred_input_activity.png"
    fig.savefig(bars_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ----- mean time courses -----
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), squeeze=False, sharex=True)
    for ax, area in zip(axes[:, 0], areas):
        co = co_by_area[area]
        mean_ts = co.mean(axis=0)  # (T, co_dim)
        co_dim = mean_ts.shape[1]
        cmap = plt.get_cmap("tab10" if co_dim <= 10 else "tab20")
        for d in range(co_dim):
            ax.plot(
                mean_ts[:, d],
                color=cmap(d % cmap.N),
                linewidth=1.3,
                label=f"d={d}",
            )
        ax.set_ylabel(f"{area}\nmean co")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncols=min(co_dim, 6), loc="upper right")
    axes[-1, 0].set_xlabel("time")
    fig.suptitle(f"{run_name}\ninferred-input mean time courses", fontsize=11)
    fig.tight_layout()
    ts_path = output_dir / f"{run_name}_inferred_input_timecourse.png"
    fig.savefig(ts_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return bars_path, ts_path


def resolve_input_h5(run_dir: Path, input_h5: Path | None) -> Path:
    if input_h5 is not None:
        path = input_h5.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Missing input H5: {path}")
        if path.name != "data.h5":
            raise SystemExit(
                f"--input-h5 must be named data.h5 (got {path.name}); "
                "BasicDataModule always appends data.h5 under datapath_override/filename."
            )
        return path

    from evals.eval_utils import resolve_mrlfads_run_data_h5

    try:
        return resolve_mrlfads_run_data_h5(run_dir)
    except Exception as exc:
        raise SystemExit(
            f"Could not resolve training data.h5 from run config ({exc}). "
            "Pass --input-h5 explicitly."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="MR-LFADS run directory (configs/main.yaml + lightning_checkpoints/).",
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=None,
        help="Observed activity HDF5 named data.h5. Default: training path from run config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs / NPZ (default: <run-dir>/inferred_input_plots).",
    )
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save co arrays to <output-dir>/<run>_inferred_inputs.npz.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    _setup_paths(repo_root)
    _patch_torch_load()

    import mrlfads.datamodules as mdm
    import mrlfads.paths as mpaths
    from mrlfads.run import run

    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "configs" / "main.yaml"
    if not config_path.is_file():
        raise SystemExit(f"Missing config: {config_path}")

    input_h5 = resolve_input_h5(run_dir, args.input_h5)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "inferred_input_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"run-dir:    {run_dir}", flush=True)
    print(f"input-h5:   {input_h5}", flush=True)
    print(f"output-dir: {output_dir}", flush=True)

    mpaths.datapath = str(repo_root / "datasets")
    mdm.path.datapath = mpaths.datapath

    model, datamodule, _ckpt = run(
        config_path=str(config_path),
        train=False,
        checkpoint_dir=str(run_dir),
    )

    datamodule.hparams.datapath_override = str(input_h5.parent)
    datamodule.hparams.filename = ""
    datamodule.hparams.p_split = [0.0, 1.0]
    datamodule.setup()

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.validate(model, datamodule=datamodule, verbose=False)

    print("Extracting inferred inputs...", flush=True)
    co_by_area = extract_inferred_inputs(model)
    if not co_by_area:
        raise SystemExit("No inferred inputs found (use_con / co_dim?).")

    bars_path, ts_path = plot_inferred_inputs(
        co_by_area, output_dir, run_name=run_dir.name, dpi=args.dpi
    )
    print(f"Wrote {bars_path}", flush=True)
    print(f"Wrote {ts_path}", flush=True)

    if args.save_npz:
        npz_path = output_dir / f"{run_dir.name}_inferred_inputs.npz"
        np.savez_compressed(npz_path, **{f"co_{a}": v for a, v in co_by_area.items()})
        print(f"Wrote {npz_path}", flush=True)

    # Quick text summary of active channels
    print("\n=== Channel RMS (active-ness) ===", flush=True)
    for area, co in co_by_area.items():
        act = channel_activity(co)
        ranked = np.argsort(-act)
        summary = ", ".join(f"d{i}={act[i]:.3f}" for i in ranked)
        print(f"  {area}: {summary}", flush=True)


if __name__ == "__main__":
    main()
