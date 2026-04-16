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
import seaborn as sns
import networkx as nx
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from itertools import permutations

import utils.visualization_utils as vis
from utils.torch_utils import RNNChannel
from utils.common_utils import (
    Messages,
    get_insert_func,
    SAVE_DIR,
    sigmoid,
    normalize,
    one_hot_encode,
)

pi = np.pi

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