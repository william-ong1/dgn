"""
Task-Trained Data Generating Networks (TT-DGN).

- Cognitive neuroscience tasks rewritten from
    Yang et al., "Task representations in neural networks trained to perform many cognitive tasks" tasks.
""" 

import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import permutations
from torch.utils.data import DataLoader, Dataset, random_split

import dgn.utils.visualization_utils as vis
from dgn.utils.torch_utils import RNNChannel, RNNBase, MLPBase
from dgn.datamodules import BasicDataset
from dgn.utils.common_utils import Messages, Flatten, get_insert_func
from dgn.models import VariableNoise

pi = np.pi
SAVE_DIR = "./graphs/"
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def normalize(theta):
    """ Convert any angle into the interval [-pi, pi]. """
    return np.arctan2(np.sin(theta), np.cos(theta))

def normalize_torch(theta):
    return torch.arctan2(torch.sin(theta), torch.cos(theta))

def sum_nested(arr):
    """ Sum uneven nested arrays. """
    total = 0
    for item in arr: total += sum(item)
    return total

def categorize(data, itvl, num_angles):
    """ Digitalize ``data`` into ``num_angles`` categories residing within the interval ``itvl``. """
    bins = torch.linspace(*itvl, num_angles + 1).to(data.device)
    return (torch.bucketize(data, bins) - 1).reshape(*data.shape)

def one_hot_encode(data, itvl, num_angles):
    """ One hot encode data with ``num_angles`` categories residing within the interval ``itvl``. """
    if num_angles is None: # if should infer num_angles from data
        num_angles = int(torch.max(data)) + 1
    data = categorize(data, itvl, num_angles)
    one_hot = torch.zeros(data.shape[:-1] + (num_angles,)).to(data.device)
    one_hot[torch.arange(data.shape[0])[:, None], torch.arange(data.shape[1]), data.squeeze()] = 1
    return one_hot

class Task:
    def __init__(
        self,
        T: int = 200,
    ):
        """ Parent class for all tasks. """
        self.T = T
    
    def gen_fixation(self, start, end=None):
        """ Fixation is 1. except for the period (start, end). """
        fix = np.ones((self.T, 1))
        fix[start: end] = 0. # after go cue, fixation is off
        return fix
    
    def gen_stimulus(self, itvl, start, end=None, amp=1.):
        """ Generate stimulus during the period (start, end), drawn from the interval ``itvl`` with amplitude ``amp``. """
        stim = np.zeros((self.T, 1))
        
        # draw random angle for stimulus
        # if itvl is scalar, it is no longer randomly drawn
        if isinstance(itvl, float): 
            theta = itvl 
        else: 
            theta = np.random.uniform(*itvl)
            
        stim[start: end, 0] = normalize(theta)
        return stim, theta, amp
    
    def gen_response(self, theta, start, end=None):
        """ Generate response ``theta`` during the period (start, end). """
        resp = np.zeros((self.T, 1))
        resp[start: end, 0] = normalize(theta)
        return resp
    
    def gen_single_trial(self):
        """ To be implemented in children classes. """
        raise NotImplementedError
        
    def draw(self, save=False, figname="task", n_batches=4):
        fig, axs = plt.subplots(5, n_batches, figsize=(8, 8), sharex=True, sharey=True)
        colors = ["k", "r", "b", "g", "k"]
        
        def round2(num): return str(round(num, 2))
        
        for b in range(n_batches):
            fix, (stim1, amp1), (stim2, amp2), resp, sacc = self.gen_single_trial()
            axs[0, b].plot(fix.squeeze(), color="k")
            axs[1, b].plot(stim1.squeeze(), color="r")
            axs[2, b].plot(stim2.squeeze(), color="b")
            axs[3, b].plot(resp.squeeze(), color="m")
            axs[4, b].plot(sacc.squeeze(), color="k")
            
            axs[0, b].set_title(f"Batch {b}")
            axs[1, b].set_title("A = " + round2(amp1) + ", \u03B8 = " + round2(max(abs(stim1.squeeze()))))
            axs[2, b].set_title("A = " + round2(amp2) + ", \u03B8 = " + round2(max(abs(stim2.squeeze()))))
            axs[3, b].set_title("\u03B8 = " + round2(max(abs(resp.squeeze()))))
            
            for i in range(5):
                vis.set_invisible(axs[i, b])
        
        ylabels = ["Fixation", "Stimulus 1", "Stimulus 2", "Response", "Saccade"]
        for i in range(5):
            axs[i, 0].set_ylabel(ylabels[i])
            
        plt.tight_layout()
        if save: vis.savefig(os.path.join(SAVE_DIR, f"{figname}.png"), clear=True, close=True)
    
class RTGo(Task):
    def __init__(
        self,
        hparams: dict,
        T: int = 200,
    ):
        """
        Task description:
            Fixation: fixation cue never goes off.
            Stimulus: stimulus occurs randomly at either channel 1 or 2.
            Response: should return stimulus direction, or the one opposite of it. [see hparams/anti]
            Saccade: should respond immediately after stimulus arrives.
        
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - anti: bool, whether to respond to the opposite direction of the stimulus
            T: time size
        """
        super().__init__(T)
        self.hps = hparams     
            
    def gen_single_trial(self):
        fix = self.gen_fixation(self.T) # fixation never goes off
        
        start = np.random.randint(30, 50) # start time of stimulus
        dur = np.random.randint(30, 50) # duration time of stimulus
        stim, theta, _ = self.gen_stimulus([-pi, pi], start, start+dur) # generate target stimulus
        null = np.zeros_like(stim) # generate "null" stimulus

        psi = theta + np.pi if self.hps["anti"] else theta # if response is opposite
        resp = self.gen_response(psi, start)
        sacc = self.gen_fixation(start) # as soon as stimulus arrives
        
        if np.random.randint(2): # determine target stimulus channel
            return fix, (stim, 1.), (null, 1.), resp, sacc
        else:
            return fix, (null, 1.), (stim, 1.), resp, sacc
            
class DlyGo(Task):
    def __init__(
        self,
        hparams: dict,
        T: int = 200,
    ):
        """
        Task description:
            Fixation: fixation cue goes off after a stimulus ends + delay period.
            Stimulus: stimulus occurs randomly at either channel 1 or 2.
            Response: should return stimulus direction, or the one opposite of it. [see hparams/anti]
            Saccade: should respond after fixation cue goes off.
        
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - anti: bool
            T: time size
        """
        super().__init__(T)
        self.hps = hparams
        
    def gen_single_trial(self):
        start = np.random.randint(30, 50)
        stim_dur = np.random.randint(30, 50)
        stim, theta, _ = self.gen_stimulus([-pi, pi], start, start + stim_dur)
        null = np.zeros_like(stim)

        dly_dur = np.random.randint(30, 50) # delay period after stimulus ends
        go_time = start + stim_dur + dly_dur # go time at arrives at t=150
        fix = self.gen_fixation(go_time)

        psi = np.pi + theta if self.hps["anti"] else theta
        resp = self.gen_response(psi, go_time)
        sacc = self.gen_fixation(go_time)
        
        if np.random.randint(2): # determine target stimulus channel
            return fix, (stim, 1.), (null, 1.), resp, sacc
        else:
            return fix, (null, 1.), (stim, 1.), resp, sacc
            
class CtxDM(Task):
    def __init__(
        self,
        hparams: dict,
        T: int = 200,
    ):
        """
        Task description:
            Fixation: fixation cue goes off after a stimulus ends + delay period.
            Stimulus: stimulus occurs at both channels, could be offset from one another. [see hparams/offset]
            Response: should return based on task type. [see hparams/to_choose]
            Saccade: should respond after fixation cue. 
            
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - to_choose: int, whether to choose stimulus 1 or 2, or the stronger one (3)
                - offset: bool, whether to offset the stimulus start time
            T: time size
        """
        super().__init__(T)
        self.hps = hparams
        
    def gen_single_trial(self):
        start = np.random.randint(30, 50)
        stim_dur = np.random.randint(30, 50)
        offset = np.random.randint(10, 20) * int(np.sign(np.random.randn())) if self.hps["offset"] else 0
        amp1, amp2 = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5) # stimulus amplitude
        
        # Generate stimulus 1
        stim1, theta1, _ = self.gen_stimulus([-pi, pi], start, start + stim_dur, amp=amp1)

        # Generate stimulus 2
        psi1 = pi + theta1 
        stim2, theta2, _ = self.gen_stimulus(
            [psi1-pi/4, psi1+pi/4], # so that stimulus 2 is sufficiently far from stimulus 1
            start + offset, start + stim_dur + offset, # stimulus 2 is offset
            amp=amp2,
        )

        dly_dur = np.random.randint(30, 50)
        go_time = start + stim_dur + max(0, offset) + dly_dur # go cue at most arrives at t=170
        fix = self.gen_fixation(go_time)
        sacc = self.gen_fixation(go_time)

        if self.hps["to_choose"] == 1:
            resp = self.gen_response(theta1, go_time)
        elif self.hps["to_choose"] == 2:
            resp = self.gen_response(theta2, go_time)
        else:
            if amp1 > amp2:
                resp = self.gen_response(theta1, go_time)
            else:
                resp = self.gen_response(theta2, go_time)
            
        return fix, (stim1, amp1), (stim2, amp2), resp, sacc
    
class Match(Task):
    def __init__(
        self,
        hparams: dict,
        T: int = 200,
    ):
        """
        Task description:
            Fixation: fixation cue goes off after second stimulus ends + delay period.
            Stimulus: stimulus occurs at both channels, is offset from one another
            Response: should return stimulus2 direction (or opposite). [see hparams/anti]
            Saccade: should respond after fixation cue goes off if the two stimuli match (based on condition), else fixate. [see hparams/cond]
            
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - anti: bool, to choose opposite direction or not
                - cond: str, "point" (if stim1==stim2) or "category" (if they are in same category)
            T: time size
        """
        super().__init__(T)
        self.hps = hparams
        
        if self.hps["cond"] == "point":
            self.same = self.same_value
        elif self.hps["cond"] == "category":
            self.same = self.same_category
        else:
            raise ValueError
        
    def gen_single_trial(self):
        start = np.random.randint(30, 50)
        stim_dur = 30
        offset = np.random.randint(10, 20) * int(np.sign(np.random.randn()))
        stim1, theta1, _ = self.gen_stimulus([-pi, pi], start, start + stim_dur)
        
        theta2 = theta1 if np.random.randint(2) else theta1 - pi # if theta1 matches theta2 or not
        stim2, _, _ = self.gen_stimulus(theta2, start + offset, start + stim_dur + offset)
        
        dly_dur = np.random.randint(30, 50)
        go_time = start + stim_dur + max(offset, 0) + dly_dur # at most t=150
        fix = self.gen_fixation(go_time)
        
        if self.same(theta1, theta2):
            sacc = self.gen_fixation(go_time)
            psi = theta2 if not self.hps["anti"] else pi + theta2 # whether to respond opposite or not
            resp = self.gen_response(psi, go_time)
        else:
            sacc = self.gen_fixation(self.T) # does not saccade
            psi = 0 # fixed response
            resp = self.gen_response(psi, go_time)
        
        return fix, (stim1, 1.), (stim2, 1.), resp, sacc
    
    @staticmethod
    def same_category(ang1, ang2):
        is_pos1 = normalize(ang1) > 0
        is_pos2 = normalize(ang2) > 0
        return not np.logical_xor(is_pos1, is_pos2)
    
    @staticmethod
    def same_value(ang1, ang2):
        return ang1 == ang2
    
# ====== Task map ===== #
# key is the task name [delay (if applicable) + task type + anti (if applicable) + stimulus to choose] 
# value is task class, hparam dict
task_map = {
    "rt_go": [RTGo, {"anti": False}],
    "rt_go_anti": [RTGo, {"anti": True}],
    "dly_go": [DlyGo, {"anti": False}],
    "dly_go_anti": [DlyGo, {"anti": True}],
    "ctxt_dm_1": [CtxDM, {"offset": False, "to_choose": 1}],
    "ctxt_dm_2": [CtxDM, {"offset": False, "to_choose": 2}],
    "ctxt_dm_max": [CtxDM, {"offset": False, "to_choose": 3}],
    "dly_dm_1": [CtxDM, {"offset": True, "to_choose": 1}],
    "dly_dm_2": [CtxDM, {"offset": True, "to_choose": 2}],
    "dly_dm_max": [CtxDM, {"offset": True, "to_choose": 3}],
    "dms": [Match, {"anti": False, "cond": "point"}],
    "dms_anti": [Match, {"anti": True, "cond": "point"}],
    "dmc": [Match, {"anti": False, "cond": "category"}],
    "dmc_anti": [Match, {"anti": True, "cond": "category"}],
}
# ===================== #

class MultiTask(pl.LightningDataModule):
    def __init__(
        self,
        task_names: list,
        batch_total: int,
        time_total: int,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        train_type: str = "random",
        train_type_kwargs: dict = {},
        dm_seed: int = 0,
        noise_sig: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.current_epoch = 0
        
    def setup(self, stage=None):
        hps = self.hparams
        
        # Construct all "Task" classes
        tasks = [] 
        for i, task_name in enumerate(hps.task_names):
            task_class, task_config = task_map[task_name]
            task = task_class(task_config, hps.time_total)
            tasks.append(task)
            try:
                task.draw(save=True, figname=task_name, n_batches=4)
            except:
                print('Task draw() failed. Continuing...')
        self.tasks = tasks
            
        batch_train = int(hps.batch_total * hps.p_split[0])
        batch_val = hps.batch_total - batch_train
        
        # Generate validation set
        self.val_batch_order = self.random_order(batch_val)
        self.val_ds = self.gen_data(tasks, self.val_batch_order)
        
        # Generate train set
        self.gen_train_ds(batch_train)
        
    def gen_train_ds(self, batch_train):
        hps = self.hparams
        if hps.train_type == "random":
            train_batch_order = self.random_order(batch_train)
        elif hps.train_type == "batch_uniform":
            train_batch_order = self.single_task_per_batch_order(batch_train, **hps.train_type_kwargs,)
        elif hps.train_type == "curriculum_ratio":
            train_batch_order = self.curriculum_ratio_order(batch_train, **hps.train_type_kwargs,)
        elif hps.train_type == "curriculum_interval":
            train_batch_order = self.curriculum_interval_order(batch_train, **hps.train_type_kwargs,)
        else:
            raise ValueError(f"Unknown train_type: {hps.train_type}")
            
        self.train_batch_order = train_batch_order
        self.train_ds = self.gen_data(self.tasks, self.train_batch_order)
        # self.plot_batch_order(self.train_batch_order, self.val_batch_order)
        
    def gen_data(self, tasks, batch_order):
        def gen_noise(arr, mag=1):
            return np.random.normal(0, 1, arr.shape) * mag * hps.noise_sig
        
        hps = self.hparams
        batch_size = len(batch_order)
        fixs = np.zeros((batch_size, hps.time_total, 1))
        stim1s = np.zeros((batch_size, hps.time_total, 1))
        stim2s = np.zeros((batch_size, hps.time_total, 1))
        resps = np.zeros((batch_size, hps.time_total, 1))
        saccs = np.zeros((batch_size, hps.time_total, 1))
        amp1s = np.zeros((batch_size, 1))
        amp2s = np.zeros((batch_size, 1))
        
        for b, tpe in enumerate(batch_order):
            fix, (stim1, amp1), (stim2, amp2), resp, sacc = tasks[int(tpe)].gen_single_trial()
            fixs[b, :] = fix + gen_noise(fix)
            stim1s[b, :] = normalize(stim1 + gen_noise(stim1, mag=0.1)) # stim1
            stim2s[b, :] = normalize(stim2 + gen_noise(stim2, mag=0.1)) # stim2
            resps[b, :] = resp
            saccs[b, :] = sacc
            amp1s[b] = amp1
            amp2s[b] = amp2
            
        task_idxs = np.tile(batch_order.reshape(-1, 1, 1), (1, hps.time_total, 1))
        return BasicDataset(fixs, stim1s, amp1s, stim2s, amp2s, task_idxs, resps, saccs)
            
    def random_order(self, batch_size, **kwargs):
        """ Ensures that each task goes once before another task goes again. """
        hps = self.hparams
        num_task = len(hps.task_names)
        task_idxs = np.arange(num_task).astype(int)
        
        rng = np.random.default_rng(hps.dm_seed)
        num_mini_batch = int(np.ceil(batch_size/num_task))
        batch = np.zeros(num_mini_batch * num_task)
        for b in range(num_mini_batch):
            rng.shuffle(task_idxs)
            batch[b * num_task: (b+1) * num_task] = task_idxs
        
        return batch[:batch_size]
    
    def single_task_per_batch_order(self, batch_total, **kwargs):
        defaults = {'persist': 1}
        defaults.update(kwargs)
        persist = defaults['persist']
        
        hps = self.hparams
        num_task = len(hps.task_names)
        task_idxs = np.arange(num_task).astype(int)
        
        base = np.tile(task_idxs.reshape(1, -1), (hps.batch_size * persist, 1)) # shape = (bs & persist, num_tasks)
        base = base.flatten(order='F') # [0, 0, 0... 1, 1, 1....,]
        num_repeats = int(np.ceil(batch_total / len(base)))
        return np.tile(base, num_repeats)[:batch_total]
    
    def curriculum_ratio_order(self, batch_total: int, **kwargs):
        defaults = {
            "persist": 1,
            "final_ratio": None,   # if None -> stays uniform
            "decay": 50.0,
        }
        defaults.update(kwargs)
        persist     = int(defaults["persist"])
        final_ratio = defaults["final_ratio"]
        decay       = float(defaults["decay"])

        hps = self.hparams
        num_task = len(hps.task_names)
        epoch = getattr(self, "current_epoch", 0)

        # Define initial (uniform) and target ratios
        init_ratio = np.ones(num_task, dtype=float) / num_task

        if final_ratio is None:
            target_ratio = init_ratio.copy()
        else:
            target_ratio = np.asarray(final_ratio, dtype=float)
            assert target_ratio.shape[0] == num_task, \
                f"final_ratio length {target_ratio.shape[0]} must match num_tasks={num_task}"
            target_ratio = target_ratio / target_ratio.sum()

        # alpha ~ 0 => uniform, alpha ~ 1 => target_ratio
        if decay <= 0:
            alpha = 1.0
        else:
            alpha = 1.0 - np.exp(-epoch / decay)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        curr_ratio = (1.0 - alpha) * init_ratio + alpha * target_ratio
        curr_ratio = curr_ratio / curr_ratio.sum()

        # Sample block-wise tasks according to curr_ratio
        block_size = hps.batch_size * persist
        num_blocks = int(np.ceil(batch_total / block_size))

        # Make RNG depend on epoch so pattern changes across epochs
        rng = np.random.default_rng(hps.dm_seed + int(epoch))

        # Choose a task id for each block
        block_tasks = rng.choice(num_task, size=num_blocks, p=curr_ratio)
        batch = np.repeat(block_tasks, block_size)
        return batch[:batch_total]
    
    def curriculum_interval_order(self, batch_total: int, **kwargs):
        raise NotImplementedError
    
    def train_dataloader(self, shuffle=False):
        return DataLoader(
            self.train_ds,
            batch_size = self.hparams.batch_size,
            shuffle=False, # to accomodate for train_type
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size = len(self.val_ds),
            shuffle=False,
            num_workers=16,
        )
    
    def plot_batch_order(self, order1, order2):
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(order1, color='red', linestyle='', marker='.', markersize=1)
        axs[1].plot(order2, color='blue', linestyle='', marker='.', markersize=1)
        vis.common_label(fig, 'epochs', 'task type')
        plt.savefig(os.path.join(SAVE_DIR, f'batch_order_epoch={self.current_epoch}.png'), dpi=300)
            
class MultiTaskNet(pl.LightningModule):
    """Multi-regional RNNs that perform multiple tasks jointly."""
    def __init__(
        self,
        num_areas: int,
        task_names: list,
        hidden_size: int = 32,
        num_angles: int = 36,
        num_channels: int = 4,
        noise: float = 0.0,
        noise_type: str = 'fixed',
        channel_noise: float = 0.0,
        channel_noise_type: str = 'fixed',
        lr_init: float = 4.0e-3,
        angle_start_epoch: int = 50,
        angle_increase_epoch: int = 100,
        angle_scale: float = 0.0,
        l1_start_epoch: int = 150,
        l1_increase_epoch: int = 150,
        l1_scale: float = 0.0,
        delay: int = 0,
        diagram: list = None,
        sacc_scale: float = 1.0,
        sacc_output_areas: list = None,   # areas that are required to saccade, defaults to output area only
        stim_input_areas: list = None,    # areas that receive inputs=(fix, stim1, stim2), defaults to A0 if unspecified
        graph_kwargs: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["task_names"])
        hps = self.hparams
        hps.task_names = task_names
        hps.num_tasks = len(hps.task_names)
        
        if not stim_input_areas: hps.stim_input_areas = ["A0"] * 3
        else: hps.stim_input_areas = stim_input_areas
        
        self._build_areas()
        self.insert_func, self.slice_func = self.get_insert_func_nested(hps.total_mesgs)
        self.mseloss = nn.MSELoss(reduction="none")
        self.bceloss = nn.BCELoss(reduction="none")
        self.celoss = nn.CrossEntropyLoss(reduction="none")
        
        # Get indices corresponding to the input of an area
        # key of input_indices is the area_name
        # values of input_indices is a list of tuples, (source_area, index of area_name in source_area's outputs)
        nested_idxs = {area_name: [] for area_name in self.area_names + [f"A{hps.num_areas}"]}
        for ia, area_name in enumerate(self.area_names + [f"A{hps.num_areas}"]):
            src_idxs = [int(node[1:]) for node in self.G.predecessors(area_name)]
            for src_idx in src_idxs:
                decs = sorted([int(node[1:]) for node in self.G.successors(f"A{src_idx}")])
                nested_idxs[area_name].append((src_idx, decs.index(ia)))
        self.input_indices = nested_idxs
        
        # Build mask
        edges = {}
        for src, tar, data in self.G.edges(data=True):
            if tar == f"A{hps.num_areas}":
                if int(src[1:]) not in edges.keys(): edges[int(src[1:])] = []
                edges[int(src[1:])].append(int(data["weight"])-1) # minus 1 resets the index to match task
        self.edges = edges
        
        # Build variable noise update class
        assert self.hparams.noise_type in ['fixed', 'variable']
        self.noise_weight_generator = VariableNoise(
            self.hparams.noise,
            hps.num_areas * hps.hidden_size,
            device = 'cuda',
            off = (self.hparams.noise_type == 'fixed'),
        )
        assert self.hparams.channel_noise_type in ['fixed', 'variable']
        self.cnoise_weight_generator = VariableNoise(
            self.hparams.channel_noise,
            sum_nested(hps.total_mesgs),
            device = 'cuda',
            off = (self.hparams.channel_noise_type == 'fixed'),
        )
        self.h_noise_weight = torch.ones(hps.hidden_size, hps.num_areas).to(self.device) * hps.noise
        self.c_noise_weight = torch.ones(sum_nested(hps.total_mesgs)).to(self.device) * hps.channel_noise
        
    def forward(self, inp, step_type):
        hps = self.hparams
        batch, time, _ = inp[0].shape
        self._build_save_var(batch, time)
        
        h = torch.zeros(batch, hps.hidden_size * hps.num_areas).to(self.device)
        mesgs = torch.zeros(batch, sum_nested(hps.total_mesgs)).to(self.device)
        
        # Find task corresponding to each batch, shape = (batch,)
        task_idx = inp[3][:, 0, 0]
        
        self.outputs = []
        for t in range(time):
            
            # Add perturbation to hidden states
            # This is correct because h is shaped in the order of (hs of area 1, hs of area 2,...)
            # h = h + torch.randn_like(h) * torch.Tensor([hps.noise]).to(self.device)
            h = h + torch.randn_like(h) * self.h_noise_weight.reshape(1, -1).to(self.device)
            
            # Forward pass through each area
            h_ias, fixs = [], []
            mesgs_new = torch.zeros(batch, sum_nested(hps.total_mesgs)).to(self.device)
            
            for ia, (area_name, area) in enumerate(self.areas.items()):
                # Gather external input
                if area_name in hps.stim_input_areas:
                    inp_idxs = [i for i, value in enumerate(hps.stim_input_areas) if value == area_name]
                    inp_ia = [inp[inp_idx][:, t] for inp_idx in inp_idxs] 
                else:
                    inp_ia = []
                
                # Save external input
                if len(inp_ia) > 0:
                    self.inputs[area_name][:, t] = torch.cat(inp_ia, dim=1) # save input
                  
                # Gather upstream input
                for idx in self.input_indices[area_name]:
                    inp_ia.append(self.slice_func(mesgs, *idx))
                inp_ia = torch.cat(inp_ia, dim=1).to(torch.float32)
                
                # Forward pass through area and store hidden states
                h_ia, mesg_ias = area(inp_ia, h[:, ia*hps.hidden_size: (ia+1)*hps.hidden_size])
                h_ias.append(h_ia)
                
                # Store fixation 
                in_output = (area_name in ["A" + str(tup[0]) for tup in self.input_indices[f"A{hps.num_areas}"]])
                if len(hps.sacc_output_areas) > 0:
                    if area_name in hps.sacc_output_areas:
                        sacc_mask = torch.ones_like(mesg_ias[0]).to(self.device)
                    else:
                        sacc_mask = torch.zeros_like(mesg_ias[0]).to(self.device)
                
                else:
                    sacc_mask = torch.ones_like(mesg_ias[0]).to(self.device)
                fixs.append(mesg_ias[0] * sacc_mask)
                
                # Store messages and hidden states
                for im, mesg_ia in enumerate(mesg_ias[1:]):
                    self.insert_func(mesgs_new, mesg_ia, ia, im)
                self.hidden_states[area_name][:, t] = h_ia
                    
            # Forward pass through output area
            inp_ia, mask = [], []
            for idx in self.input_indices[f"A{hps.num_areas}"]:
                
                # mask the input from each area according to whether their designated task matched current task
                mask = (task_idx.unsqueeze(1) == torch.Tensor(self.edges[idx[0]]).to(self.device)).any(dim=1).to(int)
                output = self.slice_func(mesgs, *idx)
                inp_ia.append(output * mask.reshape(-1, 1))
                
            inp_ia = torch.cat(inp_ia, dim=1)
            output = self.output_area(inp_ia)
            self.outputs.append( output.unsqueeze(1) )
            
            # Save and reset
            self.save_var.mesgs[:, t] = mesgs_new
            self.save_var.latents[:, t] = torch.cat(fixs, dim=-1)
            self.projs[:, t] = self.readout(output)
            h = torch.cat(h_ias, dim=-1)
            
            # Introduce delays
            if t >= hps.delay:
                mesgs = self.save_var.mesgs[:, t-hps.delay]
            else:
                pass
            
            # Add noise to messages (shouldn't affect save_var)
            channel_noise = torch.randn_like(mesgs) * self.c_noise_weight.reshape(1, -1).to(self.device)
            mesgs = mesgs + channel_noise
            
        # Adjust noise weight
        self.h_noise_weight = self.noise_weight_generator(torch.cat([hs for hs in self.hidden_states.values()], dim=2)) # hidden states from all areas
        self.c_noise_weight = self.cnoise_weight_generator(self.save_var.mesgs)
        # Outputs 
        self.outputs = torch.cat(self.outputs, dim=1)
        return self.outputs
            
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        inp, info = batch
        self.current_batch = inp
        self.current_info = info
        
        # ===== One-hot encode and forward ===== #
        fix, stim1, amp1, stim2, amp2, task, resp, sacc = inp
        resp = resp.float()
        polar1 = torch.cat([stim1, torch.tile(amp1.unsqueeze(1), (1, stim1.shape[1], 1))], dim=2)
        polar2 = torch.cat([stim2, torch.tile(amp2.unsqueeze(1), (1, stim2.shape[1], 1))], dim=2)
        self.forward([fix, polar1, polar2, task], step_type)
        
        # ===== Fixation loss ===== #
        loss_sacc = self.bceloss(torch.sigmoid(self.save_var.latents), torch.tile(sacc, (1, 1, hps.num_areas)).float())
        loss_sacc = torch.mean(loss_sacc) * hps.sacc_scale
        
        # ===== Response loss ===== #
        # Compute ramp
        angle_ramp = self._compute_ramp(hps.angle_start_epoch, hps.angle_increase_epoch)
        
        # Get cosine, sine values
        cosine_pred = torch.cos(self.projs)
        sine_pred = torch.sin(self.projs)
        cosine_true = torch.cos(resp).to(self.device)
        sine_true = torch.sin(resp).to(self.device)
        
        # One hot encode target response and outputs
        resp_onehot = one_hot_encode(normalize_torch(resp), [-pi, pi], hps.num_angles)
        proj_onehot = one_hot_encode(normalize_torch(self.projs), [-pi, pi], hps.num_angles)
        
        # Get resp_mask (assumes response angle is not exactly zero)
        resp_mask = torch.where(resp != 0, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) # shape = (batch, time, 1)
        
        # Get angle (mse) loss
        loss_angle = self.mseloss(cosine_true, cosine_pred) + self.mseloss(sine_true, sine_pred)
        loss_angle = torch.mean(loss_angle * resp_mask) * hps.angle_scale
        
        # Get response (cross entropy) loss
        flatten = lambda arr: arr.reshape(-1, arr.shape[-1])
        loss_resp = self.celoss(flatten(self.outputs), flatten(resp_onehot)).reshape(*resp.shape)
        loss_resp = torch.mean(loss_resp * resp_mask)
        
        # Get response loss (mse) outside response period
        loss_base = self.mseloss(self.outputs, torch.zeros_like(self.outputs).to(self.device))
        base_mask = torch.where(resp == 0, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
        loss_base = torch.mean(loss_base * base_mask)
        
        # ===== Regularization loss ===== #
        # Ramp
        l1_ramp = self._compute_ramp(hps.l1_start_epoch, hps.l1_increase_epoch)
        
        # Get all communication layers
        linear_weights = []
        for area_name, area in self.areas.items():
            start, end = hps.successor_list[area_name]
            for layer in area.output.model:
                if isinstance(layer, nn.Linear):
                    linear_weights.append((layer.weight[start:end], hps.l1_scale))
            
        # Calculate l1 loss
        loss_l1, kernel_size = 0.0, 0
        for kernel, weight in linear_weights:
            if weight > 0:
                loss_l1 += weight * torch.norm(kernel, 1)
                kernel_size += kernel.numel()
        loss_l1 /= kernel_size + 1e-8
            
        # ===== Calculate accuracy ===== # (only using the last time point)
        pred   = torch.argmax(self.outputs[:, -1].detach().cpu(), dim=-1)   # (batch,)
        target = torch.argmax(resp_onehot[:, -1].detach().cpu(), dim=-1)    # (batch,)
        acc    = (pred == target).float()                                   # (batch,)

        per_task_acc = {}   # only tasks actually present in this batch

        # task: (batch, time, 1)
        task_last = task[:, -1, 0].detach().cpu().long()   # (batch,)

        for itask, task_name in enumerate(hps.task_names):
            # boolean mask for samples of this task in the batch (1D, matches acc)
            mask = (task_last == itask)                    # (batch,)

            if mask.any():                                 # skip if no samples of this task
                task_acc = acc[mask].mean().item()
                per_task_acc[task_name] = task_acc
        
        # Get total loss and save
        loss = loss_sacc + loss_resp + loss_base + loss_angle * angle_ramp + loss_l1 * l1_ramp
        metrics = {
            f"{step_type}/loss": loss,
            f"{step_type}/loss_sacc": loss_sacc,
            f"{step_type}/loss_resp": loss_resp,
            f"{step_type}/loss_base": loss_base,
            f"{step_type}/loss_angle": loss_angle,
            f"{step_type}/loss_l1": loss_l1,
        }
        
        for task_name, acc_val in per_task_acc.items():
            metrics[f"{step_type}/acc_{task_name}"] = acc_val
        
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=inp[0].shape[0],
        )
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        
        if batch_idx == 0:
            try:
                fig, ax = plt.subplots(1, 1)
                self.draw(ax)
                vis.savefig(f"{SAVE_DIR}network.png")
            except:
                pass
        
        return self._shared_step(batch, "valid")
    
    def predict_step(self, batch, batch_idx, step_type="valid"):
        return self._shared_step(batch, step_type)
    
    def configure_optimizers(self):
        hps = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = hps.lr_init,
        )
        return optimizer
    
    def _build_graph(self): # DEPRECIATED
        hps = self.hparams
        self.area_names = [f"A{ia}" for ia in range(hps.num_areas)]
        all_conns = list(permutations(self.area_names, 2))
        
        # Sub-output area (A{num_areas-1}) must be within 2 steps of input areas
        stop_search, iters = False, 0
        while (not stop_search) and (iters <= 500):
            random.shuffle(all_conns)
            edges = all_conns[:int(hps.graph_kwargs["perc_conns"] * len(all_conns))]
            
            stop_search = True
            graph = nx.DiGraph(edges)
            graph.add_nodes_from(self.area_names)
            for inp_area_name in hps.stim_input_areas:
                shortest = nx.shortest_path(graph, source=inp_area_name, target=f"A{hps.num_areas-1}")
                if len(shortest)-1 > hps.graph_kwargs["shortest"]:
                    stop_search = False
                    break
            iters += 1
            
            if len(self.area_names) <= 3:
                stop_search = True
            
        if not stop_search: raise RuntimeError("Graph search exceeded max iters.")
        
        # Make edges weighted
        wedges = []
        for source, target in edges:
            wedges.append((source, target, 0))
        
        # Assign connection from sub-output to output area
        wedges.append((f"A{hps.num_areas-1}", f"A{hps.num_areas}", 1))
                
        # Draw graph
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(wedges)
        self.edges = wedges
        self.G = graph
        return graph
    
    def _build_specified_graph(self):
        hps = self.hparams
        self.area_names = [f"A{ia}" for ia in range(hps.num_areas)]
        wedges = hps.diagram
        
        # Draw graph
        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(wedges)
        self.edges = wedges
        self.G = graph
        return graph
        
    def _build_areas(self):
        hps = self.hparams
        self.areas = nn.ModuleDict()
        
        if hps.diagram:
            G = self._build_specified_graph()
        else:
            G = self._build_graph()
            
        # Handling output area
        output_name = f"A{hps.num_areas}"
        sG = nx.ego_graph(G, f"A{hps.num_areas}", undirected=True) # output-related graph
        rG = G.copy()
        rG.remove_node(f"A{hps.num_areas}") # remaining graph
        
        # Build regular areas
        hps.total_mesgs = []
        hps.successor_list = {}
        for ia in range(hps.num_areas):
            area_name = f"A{ia}"
            
            # Get number of inputs
            num_input = 0
            if area_name not in hps.stim_input_areas: pass
            else:
                if hps.stim_input_areas[0] == area_name: num_input += 1 # fix
                if hps.stim_input_areas[1] == area_name: num_input += 2 # stim1
                if hps.stim_input_areas[2] == area_name: num_input += 2 # stim2
                if hps.stim_input_areas[3] == area_name: num_input += 1 # task
            
            # Get upstream and downstream channels
            num_up = len(list(rG.predecessors(f"A{ia}"))) * hps.num_channels
            num_down = [hps.num_channels] * len(list(rG.successors(f"A{ia}")))
            
            if (f"A{ia}", f"A{hps.num_areas}") in sG.edges:
                num_out = [hps.num_angles]
                flag_out = -hps.num_angles
            else:
                num_out = []
                flag_out = -1
            
            # output: fixation (1) + number downstream + number output
            self.areas[f"A{ia}"] = RNNChannel(
                num_input + num_up,
                hps.hidden_size,
                [1] + num_down + num_out,
                None,
                override_single=True,
            )
            hps.total_mesgs.append(num_down + num_out)
            hps.successor_list[area_name] = (1, flag_out)
            
        # Build output area
        num_up = len(list(sG.predecessors(f"A{hps.num_areas}")))
        self.output_area = nn.Linear(num_up * hps.num_angles, hps.num_angles)
        
        # Build readout
        self.readout = nn.Sequential(
            nn.Linear(hps.num_angles, 1),
        )
    
    def _build_save_var(self, batch_size, time):
        hps = self.hparams
        # mesgs for communication, latents for fixation
        self.save_var = Messages(
            mesgs = torch.zeros(batch_size, time, sum_nested(hps.total_mesgs)).to(self.device),
            latents = torch.zeros(batch_size, time, hps.num_areas).to(self.device),
        )
        
        self.hidden_states = {}
        self.inputs = {}
        for ia, area_name in enumerate(self.area_names):
            self.hidden_states[area_name] = torch.zeros(batch_size, time, hps.hidden_size).to(self.device)
            
            num_input = 0
            if area_name not in hps.stim_input_areas: pass
            else:
                if hps.stim_input_areas[0] == area_name: num_input += 1 # fix
                if hps.stim_input_areas[1] == area_name: num_input += 2 # stim1
                if hps.stim_input_areas[2] == area_name: num_input += 2 # stim2
                if hps.stim_input_areas[3] == area_name: num_input += 1 # task
            self.inputs[area_name] = torch.zeros(batch_size, time, num_input).to(self.device)
            
        self.projs = torch.zeros(batch_size, time, 1).to(self.device)
            
    def draw(self, ax=None):
        graph = self.G.copy()
        graph.add_edge("fix", "A0")
        graph.add_edge("stim", "A1")
        graph.add_edge("task", "A2")
        
        def color_rule(node_name):
            if "A" in node_name: 
                if f"A{self.hparams.num_areas}" != node_name:
                    return "skyblue"
                else:
                    return "limegreen"
            else:
                return "salmon"
        color_map = [color_rule(node) for node in graph.nodes]
        
#         def edge_color_rule(weight):
#             if not weight: return "k"
#             else:
#                 colors =list(sns.color_palette("hls", len(hps.num_tasks)))
#                 return colors[weight-1]
            
#         edge_color_map = [edge_color_rule(weight) for src, tar, weight in graph.edges(data=True)]
        
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=800, node_color=color_map, font_weight='bold', connectionstyle='arc3, rad = 0.1', ax=ax)
            
    @staticmethod
    def get_insert_func_nested(arr):
        """ Get insert/slice functions for nested indices.
        
        Args:
            arr: nested list, where idx1 is the source area, and idx2 is the output area
        """
        arr_flat = []
        for item in arr:
            arr_flat += item
        
        insert_func, exclude_func, slice_func = get_insert_func(arr_flat, return_slice=True)

        def get_idx(idx1, idx2):
            sum_idx = 0
            for i in range(idx1):
                sum_idx += len(arr[i])
            sum_idx += len(arr[idx1][:idx2])
            return sum_idx

        def insert_wrap(tensor, data, idx1, idx2):
            converted_idx = get_idx(idx1, idx2)
            insert_func(tensor, data, converted_idx)
        
        def slice_wrap(tensor, idx1, idx2):
            converted_idx = get_idx(idx1, idx2)
            return slice_func(tensor, converted_idx)

        return insert_wrap, slice_wrap
    
    def _compute_ramp(self, start, increase):
        return self.compute_ramp_inner(self.current_epoch, start, increase)
    
    @staticmethod
    def compute_ramp_inner(epoch, start, increase):
        # Compute a coefficient that ramps from 0 to 1 over `increase` epochs
        ramp = (epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)
    
class TaskRespPlot:
    def __init__(self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
        num_batches: int = 4,
    ):
        """
        Plot task and response.
        
        Each column is a batch.
        Row 1 are the fixation target and saccade (after sigmoid).
        Row 2 are the response target (after argmax) and output (population).
        """
        self.name = "taskrespplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.num_batches = num_batches
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
        if trainer.current_epoch <= 1: return

        hps = pl_module.hparams
        num_rows, num_cols = 2, max([self.num_batches, hps.num_tasks])
        
        # Get messages to plot
        fix, stim1, amp1, stim2, amp2, task, resp, sacc = pl_module.current_batch
        resp_angles = resp.cpu().detach()
        resp = one_hot_encode(resp, [-pi, pi], hps.num_angles).cpu().detach()
        outputs = pl_module.outputs.cpu().detach()

        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*3),
            sharex = False,
            sharey = False,
        )
        
        colors = sns.color_palette("hls", hps.num_areas)
        for b in range(num_cols):
            # Plot fixation and saccade
            axs[0, b].plot(sacc[b, :, 0].cpu().detach().numpy(), "k")
            for n in range(hps.num_areas):
                axs[0, b].plot(sigmoid(pl_module.save_var.latents[b, :, n].cpu().detach().numpy()), color=colors[n], label=f"A{n}")
            
            # Plot response
            resp_class = torch.argmax(resp, dim=2).numpy()
            resp_mask = torch.where(resp_angles != 0, torch.tensor(1), torch.tensor(0)).numpy()[..., 0] # shape = (batch, time)
            angle_class = torch.argmax(outputs, dim=2).numpy()
            axs[1, b].plot(resp_class[b, :] * resp_mask[b, :], "k")
            axs[1, b].plot(angle_class[b, :] * resp_mask[b, :] + outputs[b, :].numpy().mean(axis=-1) * (1-resp_mask[b, :]), color="b", linestyle="--")
            
        axs[0, 0].legend()
        vis.common_col_title(fig, [f"Batch {i}" for i in range(num_cols)], axs.shape)
        vis.common_row_ylabel(fig, ["Fixation", "Response"], axs.shape)
        vis.savefig(f"TaskResp_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)    
        
class get_id_conditions:
    def __init__(self,):
        self.name = "id_condition"
    
    def run(self, trainer, pl_module, **kwargs):
        info_dict = kwargs["info"][0]
        task = info_dict["task"][:, 0, 0]
        
        categories, inverse_indices = np.unique(task, return_inverse=True)
        unique_indices = [np.where(inverse_indices == i)[0] for i in range(len(categories))]
        pl_module.conditions = {0: (categories, unique_indices)}
        
class EpochCounter:
    def __init__(self,
        log_every_n_epochs: int = 10,
        run_steps: list = ["train"],
        off: bool = False,
    ):
        self.name = "epochcounter"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.off = off
    
    def run(self, trainer, pl_module, **kwargs):
        if self.off: return
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
    
        dm = trainer.datamodule
        dm.current_epoch = trainer.current_epoch
        hps = dm.hparams
        batch_train = int(hps.batch_total * hps.p_split[0])
        dm.gen_train_ds(batch_train)
        
def unwrap_messages(syn):
    mesgs = {}
    for area_name in syn.area_names:
        for src, idx in syn.input_indices[area_name]:
            if "A" + str(src) in syn.area_names:
                mesgs[("A" + str(src), area_name)] = syn.slice_func(
                    syn.save_var.mesgs, src, idx
                ).cpu().detach().numpy()
    return mesgs