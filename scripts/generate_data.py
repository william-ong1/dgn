import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def save_run_configs(run_dir_path: Path, cfg_dir: Path) -> None:
    """
    Save the experiment configurations to the run directory.
    
    Args:
        run_dir_path (Path): the path to the run directory.
        cfg_dir (Path): the path to the configuration directory.
    """
    save_dir = run_dir_path / "configs"
    shutil.copytree(cfg_dir, save_dir, dirs_exist_ok=True)


def persist_realized_diagram(model, run_dir_path: Path) -> None:
    """Write the model's realized ``diagram`` into the run's saved configs.

    Random connectomes are sampled at init; eval and reruns read
    ``configs/model/model.yaml``, so the sampled edges must be stored there.
    """
    diagram = getattr(getattr(model, "hparams", None), "diagram", None)
    if not diagram:
        return
    diagram_list = [[str(src), str(dst), str(weight)] for src, dst, weight in diagram]

    OmegaConf.save(
        OmegaConf.create({"diagram": diagram_list}),
        run_dir_path / "realized_diagram.yaml",
    )

    model_yaml = run_dir_path / "configs" / "model" / "model.yaml"
    if model_yaml.is_file():
        cfg = OmegaConf.load(model_yaml)
        cfg.diagram = diagram_list
        OmegaConf.save(cfg, model_yaml)

    resolved = run_dir_path / "resolved_config.yaml"
    if resolved.is_file():
        cfg = OmegaConf.load(resolved)
        if "model" in cfg:
            cfg.model.diagram = diagram_list
            OmegaConf.save(cfg, resolved)


def get_checkpoint_path(path: str, *, anchor: Path) -> str:
    """
    Resolve a checkpoint path for ``Trainer.fit(ckpt_path=...)``.

    Relative paths are joined to ``anchor`` (use the project root so paths like
    ``runs/<run_id>/lightning_checkpoints/best.ckpt`` work regardless of ``chdir``).
    """
    p = Path(path).expanduser()
    p = p.resolve() if p.is_absolute() else (anchor / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return str(p)


def run_data_generation(
    config_path: str,
    run_dir: str,
    seed: int = 42,
    ckpt_path: str | None = None,
    overrides: list[str] | None = None,
) -> None:
    """
    Run the data generation process.

    Args:
        config_path: Path to the main Hydra YAML (its parent is the config directory).
        run_dir: Directory for this run (created if needed; also becomes the process cwd).
        seed: RNG seed (default 42 unless overridden via CLI).
        ckpt_path: Optional ``.ckpt`` for ``Trainer.fit(ckpt_path=...)``.
        overrides: Hydra-style ``key=value`` overrides, e.g.
            ``["model.hidden_size=128", "model.input_weight_init_var_scale=2.0"]``.
    """

    # Register the project root and src directory to the Python path
    # scripts/ -> repo root
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Load the configuration
    config_file = Path(config_path).expanduser().resolve()
    cfg_dir = config_file.parent
    cfg_name = config_file.stem

    overrides = list(overrides) if overrides else []
    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        config_obj = compose(config_name=cfg_name, overrides=overrides)

    # Create the run directory and save the configurations
    run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)
    save_run_configs(
        run_dir_path=run_dir_path,
        cfg_dir=cfg_dir,
    )

    OmegaConf.save(config_obj, run_dir_path / "resolved_config.yaml")
    if overrides:
        (run_dir_path / "overrides.txt").write_text("\n".join(overrides) + "\n")

    # Resolve the checkpoint path
    resolved_ckpt = (
        get_checkpoint_path(ckpt_path, anchor=project_root)
        if ckpt_path is not None
        else None
    )

    os.chdir(run_dir_path)

    pl.seed_everything(seed, workers=True)

    # Instantiate the datamodule and model
    datamodule = instantiate(config_obj.datamodule, _convert_="all")
    model = instantiate(config_obj.model)
    persist_realized_diagram(model, run_dir_path)

    # Instantiate the trainer
    trainer = instantiate(
        config_obj.trainer,
        callbacks=[instantiate(c) for c in config_obj.callbacks.values()],
        logger=[instantiate(lg) for lg in config_obj.logger.values()],
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        num_sanity_val_steps=0,
        accelerator="auto",
        devices="auto"
    )

    fit_kw = {}
    if resolved_ckpt is not None:
        fit_kw["ckpt_path"] = resolved_ckpt
    trainer.fit(model=model, datamodule=datamodule, **fit_kw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic neural data with data generating networks (DGNs).")

    parser.add_argument(
        "experiment_name",
        type=str,
        help="Name of the experiment to run. Must be a subdirectory of the `configs` directory.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run folder name. Defaults to <experiment_name>_id<timestamp>.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for Lightning and libraries (default: 42).",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a Lightning .ckpt file: absolute, or relative to the project root "
            "(the repo root, parent of scripts/). "
            "Example: runs/<run_name>/lightning_checkpoints/best.ckpt"
        ),
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Convenience override for model.hidden_size (neuron count per area).",
    )

    parser.add_argument(
        "--input_weight_init_var_scale",
        type=float,
        default=None,
        help="Convenience override for model.input_weight_init_var_scale.",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Convenience override for model.noise (e.g. MemoryNetwork / MultiTask).",
    )

    parser.add_argument(
        "--noise_p",
        type=float,
        default=None,
        help="Convenience override for model.noise_p (PassDecision pass-area noise).",
    )

    parser.add_argument(
        "--noise_d",
        type=float,
        default=None,
        help="Convenience override for model.noise_d (PassDecision decision-area noise).",
    )

    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Hydra-style override (repeatable). "
            "Example: --override model.noise=0.1 --override model.memory=4"
        ),
    )

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent

    config_path = project_root / "configs" / args.experiment_name / "main.yaml"

    # Check if the configuration path exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Build Hydra override list from convenience flags + generic --override.
    overrides: list[str] = []
    if args.hidden_size is not None:
        overrides.append(f"model.hidden_size={args.hidden_size}")
    if args.input_weight_init_var_scale is not None:
        overrides.append(
            f"model.input_weight_init_var_scale={args.input_weight_init_var_scale}"
        )
    if args.noise is not None:
        overrides.append(f"model.noise={args.noise}")
    if args.noise_p is not None:
        overrides.append(f"model.noise_p={args.noise_p}")
    if args.noise_d is not None:
        overrides.append(f"model.noise_d={args.noise_d}")
    overrides.extend(args.override)

    # Build a default run_name that encodes the swept knobs so re-runs
    # don't clobber each other.
    suffix_parts: list[str] = []
    if args.hidden_size is not None:
        suffix_parts.append(f"h{args.hidden_size}")
    if args.input_weight_init_var_scale is not None:
        suffix_parts.append(f"ws{args.input_weight_init_var_scale}")
    if args.noise is not None:
        suffix_parts.append(f"nz{args.noise}")
    if args.noise_p is not None:
        suffix_parts.append(f"nzp{args.noise_p}")
    if args.noise_d is not None:
        suffix_parts.append(f"nzd{args.noise_d}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    runs_root = project_root / "runs"
    run_name = (
        args.run_name
        or f"{args.experiment_name}{suffix}_id{datetime.now().strftime('%y%m%d%H%M')}"
    )
    run_dir = runs_root / run_name

    # Run the data generation process
    run_data_generation(
        config_path=str(config_path),
        run_dir=str(run_dir),
        seed=args.seed,
        ckpt_path=args.ckpt_path,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
