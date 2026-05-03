"""
Task-Trained Data Generating Networks (TT-DGN).

- Cognitive neuroscience tasks rewritten from
    Yang et al., "Task representations in neural networks trained to perform many cognitive tasks" tasks.
""" 

import numpy as np
import matplotlib.pyplot as plt
import os

import utils.visualization_utils as vis
from utils.common_utils import normalize


pi = np.pi
SAVE_DIR = "./graphs/"


class Task:
    """
    Parent class for all tasks. 
    """
    def __init__(self, T: int = 200):
        """
        Args:
            T: Time size.
        """
        self.T = T
    
    def gen_fixation(self, start, end=None):
        """
        Fixation is 1. except for the period (start, end).
        """
        fix = np.ones((self.T, 1))
        fix[start: end] = 0.0 # after go cue, fixation is off
        return fix
    
    def gen_stimulus(self, itvl, start, end=None, amp=1.):
        """
        Generate stimulus during the period (start, end), drawn from the interval ``itvl`` with amplitude ``amp``.
        """
        stim = np.zeros((self.T, 1))
        
        # Draw random angle for stimulus, if itvl is scalar, it is no longer randomly drawn
        if isinstance(itvl, float): theta = itvl 
        else: theta = np.random.uniform(*itvl)
            
        # Normalize the stimulus angle
        stim[start: end, 0] = normalize(theta)
        return stim, theta, amp
    
    def gen_response(self, theta, start, end=None):
        """
        Generate response ``theta`` during the period (start, end).
        """
        resp = np.zeros((self.T, 1))
        resp[start: end, 0] = normalize(theta)
        return resp
    
    def gen_single_trial(self):
        raise NotImplementedError('gen_single_trial function must be implemented in subclass!')
        
    def draw(self, save=False, figname="task", n_batches=4):
        # Plot the task, each column is a batch
        fig, axs = plt.subplots(5, n_batches, figsize=(8, 8), sharex=True, sharey=True)
        colors = ["k", "r", "b", "g", "k"]
        
        def round2(num): return str(round(num, 2))
        
        # Plot each batch
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
    """
    Fixation: fixation cue never goes off.
    Stimulus: stimulus occurs randomly at either channel 1 or 2.
    Response: should return stimulus direction, or the one opposite of it. [see hparams/anti]
    Saccade: should respond immediately after stimulus arrives.
    """
    def __init__(self, hparams: dict, T: int = 200):
        """
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - anti: bool, whether to respond to the opposite direction of the stimulus
            T: Time size.
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
    """
    Fixation: fixation cue goes off after a stimulus ends + delay period.
    Stimulus: stimulus occurs randomly at either channel 1 or 2.
    Response: should return stimulus direction, or the one opposite of it. [see hparams/anti]
    Saccade: should respond after fixation cue goes off.
    """
    def __init__(self, hparams: dict, T: int = 200):
        """
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - anti: bool, whether to respond to the opposite direction of the stimulus
            T: Time size.
        """
        super().__init__(T)
        self.hps = hparams
        
    def gen_single_trial(self):
        start = np.random.randint(30, 50) # start time of stimulus
        stim_dur = np.random.randint(30, 50) # duration time of stimulus
        stim, theta, _ = self.gen_stimulus([-pi, pi], start, start + stim_dur) # generate target stimulus
        null = np.zeros_like(stim) # generate "null" stimulus

        dly_dur = np.random.randint(30, 50) # delay period after stimulus ends
        go_time = start + stim_dur + dly_dur  # random go cue time (depends on start, stim_dur, dly_dur)
        fix = self.gen_fixation(go_time) # fixation cue goes off after stimulus ends + delay period

        psi = np.pi + theta if self.hps["anti"] else theta # if response is opposite
        resp = self.gen_response(psi, go_time) # generate response
        sacc = self.gen_fixation(go_time) # saccade after fixation cue goes off
        
        if np.random.randint(2): # determine target stimulus channel, 50% chance to choose target stimulus channel
            return fix, (stim, 1.), (null, 1.), resp, sacc
        else:
            return fix, (null, 1.), (stim, 1.), resp, sacc
            

class CtxDM(Task):
    """
    Fixation: fixation cue goes off after a stimulus ends + delay period.
    Stimulus: stimulus occurs at both channels, could be offset from one another. [see hparams/offset]
    Response: should return based on task type. [see hparams/to_choose]
    Saccade: should respond after fixation cue. 
    """
    def __init__(self, hparams: dict, T: int = 200):
        """
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - to_choose: int, whether to choose stimulus 1 or 2, or the stronger one (3)
                - offset: bool, whether to offset the stimulus start time
            T: Time size.
        """
        super().__init__(T)
        self.hps = hparams
        
    def gen_single_trial(self):
        start = np.random.randint(30, 50) # start time of stimulus
        stim_dur = np.random.randint(30, 50) # duration time of stimulus
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

        # Generate delay period
        dly_dur = np.random.randint(30, 50)
        go_time = start + stim_dur + max(0, offset) + dly_dur  # go cue after stimulus + delay (timing is random)
        fix = self.gen_fixation(go_time)
        sacc = self.gen_fixation(go_time)

        # Generate response based on task type
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
    """
    Fixation: fixation cue goes off after second stimulus ends + delay period.
    Stimulus: stimulus occurs at both channels, is offset from one another
    Response: should return stimulus2 direction (or opposite). [see hparams/anti]
    Saccade: should respond after fixation cue goes off if the two stimuli match (based on condition), else fixate. [see hparams/cond]
    """
    def __init__(self, hparams: dict, T: int = 200):
        """
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - anti: bool, to choose opposite direction or not
                - cond: str, "point" (if stim1==stim2) or "category" (if they are in same category)
            T: Time size.
        """
        super().__init__(T)
        self.hps = hparams
        
        # Determine the condition for matching
        if self.hps["cond"] == "point":
            self.same = self.same_value
        elif self.hps["cond"] == "category":
            self.same = self.same_category
        else:
            raise ValueError
        
    def gen_single_trial(self):
        start = np.random.randint(30, 50) # start time of stimulus
        stim_dur = 30 # duration time of stimulus
        offset = np.random.randint(10, 20) * int(np.sign(np.random.randn())) # offset of stimulus
        stim1, theta1, _ = self.gen_stimulus([-pi, pi], start, start + stim_dur) # generate target stimulus 1
        
        # Generate target stimulus 2
        theta2 = theta1 if np.random.randint(2) else theta1 - pi # if theta1 matches theta2 or not
        stim2, _, _ = self.gen_stimulus(theta2, start + offset, start + stim_dur + offset)
        
        # Generate delay period
        dly_dur = np.random.randint(30, 50)
        go_time = start + stim_dur + max(offset, 0) + dly_dur  # go cue after stimuli + delay (timing is random)
        fix = self.gen_fixation(go_time)
        
        # Generate response based on condition
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
        """
        Check if the two angles are in the same category.
        """
        is_pos1 = normalize(ang1) > 0
        is_pos2 = normalize(ang2) > 0
        return not np.logical_xor(is_pos1, is_pos2)
    
    @staticmethod
    def same_value(ang1, ang2):
        return ang1 == ang2
    

# Task Map, where key is the task name [delay (if applicable) + task type + anti (if applicable) + stimulus to choose] 
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
