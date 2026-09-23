import os
import shutil

from pathlib import Path
from datetime import datetime

import mrlfads.paths as path
from mrlfads.run import run

# ---------- USER DEFINED PARAMETERS -----------
CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_STR = CONFIG_DIR.parent.name
SEED_STR = CONFIG_DIR.name
EXIST_PROJECT_STR = None
OVERWRITE = isinstance(EXIST_PROJECT_STR, type(None))
# ----------------------------------------------

RUN_STR = (
    f"{PROJECT_STR}_{SEED_STR}_id{datetime.now().strftime('%y%m%d%H%M')}"
    if not EXIST_PROJECT_STR
    else EXIST_PROJECT_STR
)

RUN_DIR = Path(os.path.join(path.resultpath)) / RUN_STR

if OVERWRITE:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
else:
    print("WARNING: RUN_DIR NOT OVERWRITTEN.")

shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
obs_map = CONFIG_DIR / "observed_neurons.yaml"
if not obs_map.exists():
    obs_map = CONFIG_DIR.parent / "observed_neurons.yaml"
if obs_map.exists():
    shutil.copyfile(obs_map, RUN_DIR / "observed_neurons.yaml")

os.chdir(RUN_DIR)

run(
    config_path=str(CONFIG_DIR / "main.yaml"),
    checkpoint_dir=str(RUN_DIR) if EXIST_PROJECT_STR else None,
    train=True,
)
