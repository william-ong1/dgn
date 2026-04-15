import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _register_resolvers(project_root: Path) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver(
        "relpath",
        lambda p: project_root / p,
    )


def _archive_run_configs(
    *,
    run_dir_path: Path,
    cfg_dir: Path,
) -> None:
    archive_root = run_dir_path / "configs"
    archive_root.mkdir(parents=True, exist_ok=True)

    # Copy files from the experiment config folder.
    for file_path in cfg_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(cfg_dir)
        destination_path = archive_root / rel_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, destination_path)


def run_experiment(
    config: str,
    overrides: list[str] | None = None,
    seed: int | None = None,
    run_dir: str | None = None,
) -> None:
    overrides = overrides or []
    project_root = Path(__file__).resolve().parent
    _register_resolvers(project_root)

    config_file = Path(config).expanduser().resolve()
    cfg_dir = config_file.parent
    cfg_name = config_file.stem

    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        config_obj = compose(config_name=cfg_name, overrides=overrides)

    target_run_dir = run_dir if run_dir is not None else config_obj.get("run_dir", None)
    if target_run_dir:
        run_dir_path = Path(target_run_dir).expanduser().resolve()
        run_dir_path.mkdir(parents=True, exist_ok=True)
        _archive_run_configs(
            run_dir_path=run_dir_path,
            cfg_dir=cfg_dir,
        )
        os.chdir(run_dir_path)

    exp_seed = seed if seed is not None else config_obj.get("seed", None)
    if exp_seed is not None:
        pl.seed_everything(exp_seed, workers=True)

    datamodule = instantiate(config_obj.datamodule, _convert_="all")
    model = instantiate(config_obj.model)

    trainer = instantiate(
        config_obj.trainer,
        callbacks=[instantiate(c) for c in config_obj.callbacks.values()],
        logger=[instantiate(lg) for lg in config_obj.logger.values()],
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        num_sanity_val_steps=0,
        # accelerator="auto",
        # devices="auto",
    )

    trainer.fit(model=model, datamodule=datamodule)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DGN synthetic data and export H5.")

    parser.add_argument(
        "experiment",
        type=str,
        help="Config folder name under configs (e.g. memory_network, pass_decision, multi_task).",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run folder name. Defaults to <experiment>_id<timestamp>.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional explicit path to main Hydra config. If set, overrides experiment folder lookup.",
    )
    
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydra overrides like trainer.max_epochs=100 model.hidden_size=128",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.config:
        user_config_path = Path(args.config).expanduser()
        if user_config_path.is_absolute():
            config_path = user_config_path.resolve()
        else:
            config_path = (project_root / user_config_path).resolve()
    else:
        config_path = project_root / "configs" / args.experiment / "main.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    runs_root = project_root / "runs"
    run_name = args.run_name or f"{args.experiment}_id{datetime.now().strftime('%y%m%d%H%M')}"
    run_dir = runs_root / run_name

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(__file__, run_dir / Path(__file__).name)
    os.chdir(run_dir)

    run_experiment(
        config=str(config_path),
        overrides=args.overrides,
        seed=args.seed,
        run_dir=str(run_dir),
    )


if __name__ == "__main__":
    main()
