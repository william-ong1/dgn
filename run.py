import os
import shutil
import hydra
import torch
import logging
import warnings
import functools
import numpy as np
import pytorch_lightning as pl

from glob import glob
from pathlib import Path
from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

import dgn.paths as path
from dgn.utils.common_utils import check_pattern

def search(
    datapath: str,
    patterns: list = [],
):
    dirs = [d for d in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, d))]
    
    library = {}
    for d in dirs:
        # Load hparams
        yamlpath = os.path.join(datapath, d)
        cond1 = os.path.exists(os.path.join(yamlpath, 'hparams.yaml'))
        cond2 = check_pattern(d, patterns)
        
        if cond1 and cond2:
            with initialize_config_dir(version_base="1.1", config_dir=yamlpath):
                config = compose(config_name='hparams')
            hparams = OmegaConf.to_container(
                config,
                resolve=True,
                throw_on_missing=True,
            )
            library[d] = hparams
    return library

def match(
    datapath: str,
    patterns: list,
    conditions: dict = {},
):
    library = search(datapath, patterns=patterns)
    
    passes = []
    for d, hps in library.items():    
        flag = True # default is pass
        for k, v in conditions.items():
            if hps[k] != v: flag = False
            
        if flag: passes.append(d)
    return passes, {p: library[p] for p in passes}