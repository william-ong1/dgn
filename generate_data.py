import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

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


def resolve_checkpoint_for_fit(
    *,
    ckpt_path: Optional[str],
    checkpoint_dir: Optional[str],
    ckpt_match: Optional[str],
) -> Optional[str]:
    """
    Pick a checkpoint path for ``Trainer.fit(..., ckpt_path=...)``.

    Lightning will restore model weights, optimizer state, epoch counter, and
    global step from the file (full training resume, not weights-only).

    Parameters
    ----------
    ckpt_path
        Absolute or relative path to a ``.ckpt`` file.
    checkpoint_dir
        Run directory (or any folder) that contains a ``lightning_checkpoints``
        subdirectory with ``*.ckpt`` files.
    ckpt_match
        If set (with ``checkpoint_dir`` only), require exactly one checkpoint
        whose filename contains this substring (e.g. ``last`` or ``epoch=000``).
        If unset, the most recently modified ``*.ckpt`` is used.
    """
    if ckpt_path and checkpoint_dir:
        raise ValueError("Use only one of ckpt_path or checkpoint_dir.")
    if ckpt_match and not checkpoint_dir:
        raise ValueError("--ckpt_match only applies with --checkpoint_dir.")
    if ckpt_path:
        path = Path(ckpt_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
        return str(path.resolve())
    if checkpoint_dir:
        base = Path(checkpoint_dir).expanduser().resolve()
        ckpt_root = base / "lightning_checkpoints"
        if not ckpt_root.is_dir():
            raise FileNotFoundError(
                f"Expected a lightning_checkpoints directory under {base}"
            )
        candidates = sorted(ckpt_root.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No .ckpt files in {ckpt_root}")
        if ckpt_match:
            matches = [c for c in candidates if ckpt_match in c.name]
            if not matches:
                raise FileNotFoundError(
                    f"No checkpoint in {ckpt_root} with {ckpt_match!r} in the filename."
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple checkpoints match {ckpt_match!r}:\n"
                    + "\n".join(str(m) for m in matches)
                )
            return str(matches[0])
        return str(candidates[-1])
    return None


def run_experiment(
    config: str,
    overrides: list[str] | None = None,
    seed: int | None = None,
    run_dir: str | None = None,
    resume: bool = False,
    ckpt_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    ckpt_match: Optional[str] = None,
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
        if resume:
            if not run_dir_path.is_dir():
                raise FileNotFoundError(
                    f"Resume requested but run directory does not exist: {run_dir_path}"
                )
        else:
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
        accelerator="auto",
        devices="auto",
    )

    resolved_ckpt = resolve_checkpoint_for_fit(
        ckpt_path=ckpt_path,
        checkpoint_dir=checkpoint_dir,
        ckpt_match=ckpt_match,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resolved_ckpt)


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

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Use an existing run directory (--run_name must match). Does not delete the "
            "run or re-archive configs. If neither --ckpt_path nor --checkpoint_dir is set, "
            "the latest lightning_checkpoints/*.ckpt under that run is used."
        ),
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a .ckpt file. Passed to Trainer.fit(ckpt_path=...) for full resume.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Directory that contains lightning_checkpoints/*.ckpt. "
            "If --ckpt_match is omitted, the most recently modified .ckpt is used."
        ),
    )

    parser.add_argument(
        "--ckpt_match",
        type=str,
        default=None,
        help=(
            "Substring of the checkpoint filename (used with --checkpoint_dir only), "
            "e.g. last or epoch=000. Must match exactly one file."
        ),
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

    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"--resume requires an existing run directory: {run_dir}"
            )
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(__file__, run_dir / Path(__file__).name)

    os.chdir(run_dir)

    ckpt_path_arg = args.ckpt_path
    checkpoint_dir_arg = args.checkpoint_dir
    if args.resume and not ckpt_path_arg and not checkpoint_dir_arg:
        checkpoint_dir_arg = str(run_dir.resolve())
    if checkpoint_dir_arg:
        checkpoint_dir_arg = str(Path(checkpoint_dir_arg).expanduser().resolve())
    if ckpt_path_arg:
        ckpt_path_arg = str(Path(ckpt_path_arg).expanduser().resolve())

    run_experiment(
        config=str(config_path),
        overrides=args.overrides,
        seed=args.seed,
        run_dir=str(run_dir),
        resume=args.resume,
        ckpt_path=ckpt_path_arg,
        checkpoint_dir=checkpoint_dir_arg,
        ckpt_match=args.ckpt_match,
    )


if __name__ == "__main__":
    main()
