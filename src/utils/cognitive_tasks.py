"""
Task-Trained Data Generating Networks (TT-DGN).

Cognitive tasks rewritten from Yang et al., "Task representations in neural
networks trained to perform many cognitive tasks."

The 20 Yang rules are registered in ``YANG20_TASKS`` / ``task_map``. Timings
are in 200-step units (not the paper's ms schedule); trial structure matches
the paper: go / RT / delay-go (±anti), DM, context-DM, delay-DM, DMS/DMC
(go and nogo). Each trial still returns
``fix, (stim1, amp1), (stim2, amp2), resp, sacc``.
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
        fix[start: end] = 0.0  # after go cue, fixation is off
        return fix

    def gen_stimulus(self, itvl, start, end=None, amp=1.):
        """
        Generate stimulus during the period (start, end), drawn from the interval ``itvl`` with amplitude ``amp``.
        """
        stim = np.zeros((self.T, 1))

        # Draw random angle for stimulus, if itvl is scalar, it is no longer randomly drawn
        if isinstance(itvl, (float, int, np.floating, np.integer)):
            theta = float(itvl)
        else:
            theta = np.random.uniform(*itvl)

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

    def assign_channel(self, stim, amp=1.):
        """Put ``stim`` on a random channel; the other channel is silent."""
        null = np.zeros_like(stim)
        if np.random.randint(2):
            return (stim, amp), (null, amp)
        return (null, amp), (stim, amp)

    def fit_epochs(self, *durations, tail=20):
        """Scale positive durations so they fit in ``T - tail``."""
        durs = [max(1, int(d)) for d in durations]
        budget = max(len(durs), self.T - int(tail))
        total = sum(durs)
        if total > budget:
            scale = budget / float(total)
            durs = [max(1, int(round(d * scale))) for d in durs]
            while sum(durs) > budget:
                i = int(np.argmax(durs))
                if durs[i] <= 1:
                    break
                durs[i] -= 1
        return durs

    def dm_amps(self):
        """Yang-style mean ± coherence so one alternative is clearly stronger."""
        mean = float(np.random.uniform(0.8, 1.2))
        coh = float(np.random.choice([0.15, 0.30, 0.45]))
        sign = float(np.random.choice([-1.0, 1.0]))
        return mean + coh * sign, mean - coh * sign

    def gen_single_trial(self):
        raise NotImplementedError('gen_single_trial function must be implemented in subclass!')

    def draw(self, save=False, figname="task", n_batches=4):
        # Plot the task, each column is a batch
        fig, axs = plt.subplots(5, n_batches, figsize=(8, 8), sharex=True, sharey=True)

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
        if save:
            vis.savefig(os.path.join(SAVE_DIR, f"{figname}.png"), clear=True, close=True)


class RTGo(Task):
    """
    Yang reactgo / reactanti.

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
        fix = self.gen_fixation(self.T)  # fixation never goes off

        start, dur = self.fit_epochs(
            np.random.randint(30, 50),
            np.random.randint(30, 50),
        )
        stim, theta, _ = self.gen_stimulus([-pi, pi], start, start + dur)
        ch1, ch2 = self.assign_channel(stim)

        psi = theta + np.pi if self.hps["anti"] else theta
        resp = self.gen_response(psi, start)
        sacc = self.gen_fixation(start)  # as soon as stimulus arrives
        return fix, ch1, ch2, resp, sacc


class FdGo(Task):
    """
    Yang fdgo / fdanti (Go / Anti).

    Stimulus comes on while fixation is still required and stays on.
    Respond only after the fixation cue turns off (inhibitory control).
    """
    def __init__(self, hparams: dict, T: int = 200):
        super().__init__(T)
        self.hps = hparams

    def gen_single_trial(self):
        stim_on, hold = self.fit_epochs(
            np.random.randint(20, 40),
            np.random.randint(40, 80),
        )
        go_time = stim_on + hold
        # Stimulus stays on through the go period (Yang: no stim_off).
        stim, theta, _ = self.gen_stimulus([-pi, pi], stim_on, None)
        ch1, ch2 = self.assign_channel(stim)

        psi = theta + np.pi if self.hps["anti"] else theta
        fix = self.gen_fixation(go_time)
        resp = self.gen_response(psi, go_time)
        sacc = self.gen_fixation(go_time)
        return fix, ch1, ch2, resp, sacc


class DlyGo(Task):
    """
    Yang delaygo / delayanti.

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
        start, stim_dur, dly_dur = self.fit_epochs(
            np.random.randint(30, 50),
            np.random.randint(30, 50),
            np.random.randint(30, 50),
        )
        stim, theta, _ = self.gen_stimulus([-pi, pi], start, start + stim_dur)
        ch1, ch2 = self.assign_channel(stim)

        go_time = start + stim_dur + dly_dur
        fix = self.gen_fixation(go_time)

        psi = np.pi + theta if self.hps["anti"] else theta
        resp = self.gen_response(psi, go_time)
        sacc = self.gen_fixation(go_time)
        return fix, ch1, ch2, resp, sacc


class CtxDM(Task):
    """
    Decision-making family (Yang dm / contextdm / delaydm / contextdelaydm / multidm).

    Two alternatives are the two stimulus channels (location + amplitude).
    Simultaneous (offset=False) or sequential (offset=True: stim1, delay, stim2).
    Response is stim1, stim2, or the stronger one [see hparams/to_choose].
    """
    def __init__(self, hparams: dict, T: int = 200):
        """
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - to_choose: int, whether to choose stimulus 1 or 2, or the stronger one (3)
                - offset: bool, whether to present the two alternatives sequentially
            T: Time size.
        """
        super().__init__(T)
        self.hps = hparams

    def _choice_theta(self, theta1, theta2, amp1, amp2):
        which = self.hps["to_choose"]
        if which == 1:
            return theta1
        if which == 2:
            return theta2
        return theta1 if amp1 > amp2 else theta2

    def gen_single_trial(self):
        if self.hps["to_choose"] == 3:
            amp1, amp2 = self.dm_amps()
        else:
            amp1, amp2 = float(np.random.uniform(0.5, 1.5)), float(np.random.uniform(0.5, 1.5))

        if self.hps["offset"]:
            t1, d1, gap, d2, post = self.fit_epochs(
                np.random.randint(20, 40),
                np.random.randint(20, 35),
                np.random.randint(20, 40),
                np.random.randint(20, 35),
                np.random.randint(15, 30),
            )
            t2 = t1 + d1 + gap
            go_time = t2 + d2 + post
            stim1, theta1, _ = self.gen_stimulus([-pi, pi], t1, t1 + d1, amp=amp1)
            psi1 = pi + theta1
            stim2, theta2, _ = self.gen_stimulus(
                [psi1 - pi / 4, psi1 + pi / 4],
                t2,
                t2 + d2,
                amp=amp2,
            )
        else:
            start, stim_dur, dly_dur = self.fit_epochs(
                np.random.randint(30, 50),
                np.random.randint(30, 50),
                np.random.randint(30, 50),
            )
            go_time = start + stim_dur + dly_dur
            stim1, theta1, _ = self.gen_stimulus([-pi, pi], start, start + stim_dur, amp=amp1)
            psi1 = pi + theta1
            stim2, theta2, _ = self.gen_stimulus(
                [psi1 - pi / 4, psi1 + pi / 4],
                start,
                start + stim_dur,
                amp=amp2,
            )

        fix = self.gen_fixation(go_time)
        sacc = self.gen_fixation(go_time)
        resp = self.gen_response(self._choice_theta(theta1, theta2, amp1, amp2), go_time)
        return fix, (stim1, amp1), (stim2, amp2), resp, sacc


class Match(Task):
    """
    Yang dmsgo / dmsnogo / dmcgo / dmcnogo.

    Stim1, delay, stim2 (sequential). Go at stim2 onset.
    Go if the pair matches (point or category); nogo flips that rule.
    """
    def __init__(self, hparams: dict, T: int = 200):
        """
        Args:
            hparams:
                Hyperparameters for this task; this dictionary contains the following (key, value) pairs:
                - cond: str, "point" (if stim1==stim2) or "category" (if they are in same category)
                - nogo: bool, if True, go on mismatch and hold on match (Yang DNMS / DNMC)
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
            raise ValueError(f"Unknown cond: {self.hps['cond']}")

    def gen_single_trial(self):
        t1, d1, gap, d2 = self.fit_epochs(
            np.random.randint(20, 40),
            30,
            np.random.randint(20, 50),
            30,
        )
        t2 = t1 + d1 + gap
        go_time = t2

        if self.hps["cond"] == "category":
            # Yang DMC uses two half-circles; sample from the same discrete set.
            cats = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * pi
            theta1 = float(np.random.choice(cats))
            if np.random.randint(2):
                same_cat = [a for a in cats if self.same_category(theta1, a)]
                theta2 = float(np.random.choice(same_cat))
            else:
                other_cat = [a for a in cats if not self.same_category(theta1, a)]
                theta2 = float(np.random.choice(other_cat))
            stim1, _, _ = self.gen_stimulus(theta1, t1, t1 + d1)
            stim2, _, _ = self.gen_stimulus(theta2, t2, t2 + d2)
        else:
            stim1, theta1, _ = self.gen_stimulus([-pi, pi], t1, t1 + d1)
            theta2 = theta1 if np.random.randint(2) else theta1 - pi
            stim2, _, _ = self.gen_stimulus(theta2, t2, t2 + d2)

        matched = bool(self.same(theta1, theta2))
        should_go = (not matched) if self.hps.get("nogo", False) else matched

        fix = self.gen_fixation(go_time)
        if should_go:
            sacc = self.gen_fixation(go_time)
            resp = self.gen_response(theta2, go_time)
        else:
            sacc = self.gen_fixation(self.T)  # hold fixation
            resp = self.gen_response(0, go_time)
            resp[:] = 0.0
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
        return abs(normalize(ang1 - ang2)) < 1e-6


# Yang et al. 20-rule set only.
task_map = {
    # Go / Anti (fdgo, reactgo, delaygo, fdanti, reactanti, delayanti)
    "fd_go": [FdGo, {"anti": False}],
    "rt_go": [RTGo, {"anti": False}],
    "dly_go": [DlyGo, {"anti": False}],
    "fd_go_anti": [FdGo, {"anti": True}],
    "rt_go_anti": [RTGo, {"anti": True}],
    "dly_go_anti": [DlyGo, {"anti": True}],
    # DM (dm1, dm2): two simultaneous alternatives, choose stronger
    "dm_1": [CtxDM, {"offset": False, "to_choose": 3}],
    "dm_2": [CtxDM, {"offset": False, "to_choose": 3}],
    # Context / multi-sensory DM (contextdm1, contextdm2, multidm)
    "ctxt_dm_1": [CtxDM, {"offset": False, "to_choose": 1}],
    "ctxt_dm_2": [CtxDM, {"offset": False, "to_choose": 2}],
    "ctxt_dm_max": [CtxDM, {"offset": False, "to_choose": 3}],
    # Delay DM (delaydm1, delaydm2, contextdelaydm1/2, multidelaydm)
    "dly_dm_mod_1": [CtxDM, {"offset": True, "to_choose": 3}],
    "dly_dm_mod_2": [CtxDM, {"offset": True, "to_choose": 3}],
    "dly_dm_1": [CtxDM, {"offset": True, "to_choose": 1}],
    "dly_dm_2": [CtxDM, {"offset": True, "to_choose": 2}],
    "dly_dm_max": [CtxDM, {"offset": True, "to_choose": 3}],
    # Delayed match (dmsgo, dmsnogo, dmcgo, dmcnogo)
    "dms": [Match, {"cond": "point", "nogo": False}],
    "dms_nogo": [Match, {"cond": "point", "nogo": True}],
    "dmc": [Match, {"cond": "category", "nogo": False}],
    "dmc_nogo": [Match, {"cond": "category", "nogo": True}],
}

# Yang et al. 20-rule set, paper order, using this repo's names.
YANG20_TASKS = (
    "fd_go",          # fdgo
    "rt_go",          # reactgo
    "dly_go",         # delaygo
    "fd_go_anti",     # fdanti
    "rt_go_anti",     # reactanti
    "dly_go_anti",    # delayanti
    "dm_1",           # dm1
    "dm_2",           # dm2
    "ctxt_dm_1",      # contextdm1
    "ctxt_dm_2",      # contextdm2
    "ctxt_dm_max",    # multidm
    "dly_dm_mod_1",   # delaydm1
    "dly_dm_mod_2",   # delaydm2
    "dly_dm_1",       # contextdelaydm1
    "dly_dm_2",       # contextdelaydm2
    "dly_dm_max",     # multidelaydm
    "dms",            # dmsgo
    "dms_nogo",       # dmsnogo
    "dmc",            # dmcgo
    "dmc_nogo",       # dmcnogo
)
