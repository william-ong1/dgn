import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def save_run_configs(run_dir_path: Path, cfg_dir: Path) -> None:
    """
    Save the experiment configurations to the run directory.
    
    Args:
        run_dir_path (Path): the path to the run directory.
        cfg_dir (Path): the path to the configuration directory.
    """
    save_dir = run_dir_path / "configs"
    shutil.copytree(cfg_dir, save_dir, dirs_exist_ok=True)


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
) -> None:
    """
    Run the data generation process.

    Args:
        config_path: Path to the main Hydra YAML (its parent is the config directory).
        run_dir: Directory for this run (created if needed; also becomes the process cwd).
        seed: RNG seed (default 42 unless overridden via CLI).
        ckpt_path: Optional ``.ckpt`` for ``Trainer.fit(ckpt_path=...)``.
    """

    # Register the project root and src directory to the Python path
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Load the configuration
    config_file = Path(config_path).expanduser().resolve()
    cfg_dir = config_file.parent
    cfg_name = config_file.stem

    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        config_obj = compose(config_name=cfg_name)

    # Create the run directory and save the configurations
    run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)
    save_run_configs(
        run_dir_path=run_dir_path,
        cfg_dir=cfg_dir,
    )

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
            "(the directory that contains generate_data.py). "
            "Example: runs/<run_name>/lightning_checkpoints/best.ckpt"
        ),
    )

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent

    config_path = project_root / "configs" / args.experiment_name / "main.yaml"

    # Check if the configuration path exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Get the run directory
    runs_root = project_root / "runs"
    run_name = args.run_name or f"{args.experiment_name}_id{datetime.now().strftime('%y%m%d%H%M')}"
    run_dir = runs_root / run_name

    # Run the data generation process
    run_data_generation(
        config_path=str(config_path),
        run_dir=str(run_dir),
        seed=args.seed,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    main()
