"""
Task-Trained Data Generating Networks (TT-DGN).

Cognitive tasks adapted from Yang et al., "Task representations in neural
networks trained to perform many cognitive tasks."

The 20 rules are registered in ``YANG20_TASKS`` / ``task_map``. Each trial is
200 steps and returns ``fix, (stim1, amp1), (stim2, amp2), resp, sacc``.

Each stimulus modality holds two directional alternatives:
``stim`` is ``(T, 2)`` angles and ``amp`` is ``(T, 2)`` strengths (0 = silent).
Epoch lengths are drawn at random and scaled to leave a response tail.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import utils.visualization_utils as vis
from utils.common_utils import normalize


pi = np.pi
SAVE_DIR = "./graphs/"
N_ALTS = 2


class Task:
    """Parent class for all tasks."""

    def __init__(self, T: int = 200):
        self.T = T

    def gen_fixation(self, start, end=None):
        """Fixation is 1 except on ``[start, end)``."""
        fix = np.ones((self.T, 1))
        fix[start:end] = 0.0
        return fix

    def gen_response(self, theta, start, end=None):
        """Response direction ``theta`` on ``[start, end)``."""
        resp = np.zeros((self.T, 1))
        resp[start:end, 0] = normalize(theta)
        return resp

    def blank_modality(self):
        """Silent modality: two alternatives, zero angle and strength."""
        return np.zeros((self.T, N_ALTS)), np.zeros((self.T, N_ALTS))

    def write_pulse(self, stim, amp, slot, theta, strength, start, end):
        stim[start:end, slot] = normalize(theta)
        amp[start:end, slot] = float(strength)

    def assign_modality(self, stim, amp):
        """Place one modality's pulses on a random channel; the other is silent."""
        silent_s, silent_a = self.blank_modality()
        if np.random.randint(2):
            return (stim, amp), (silent_s, silent_a)
        return (silent_s, silent_a), (stim, amp)

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
        """Mean ± coherence so one alternative is clearly stronger."""
        mean = float(np.random.uniform(0.8, 1.2))
        coh = float(np.random.choice([0.15, 0.30, 0.45]))
        sign = float(np.random.choice([-1.0, 1.0]))
        return mean + coh * sign, mean - coh * sign

    def sample_direction(self):
        """Direction whose angle and opposite are both away from 0.

        A response of exactly 0 is the no-go mask in the multitask loss.
        """
        for _ in range(30):
            theta = float(normalize(np.random.uniform(-pi, pi)))
            if abs(theta) > 0.05 and abs(normalize(theta + pi)) > 0.05:
                return theta
        return 0.5

    def two_directions(self):
        """Two well-separated directions on the circle."""
        theta_a = self.sample_direction()
        theta_b = float(normalize(theta_a + pi + np.random.uniform(-pi / 4, pi / 4)))
        if abs(theta_b) <= 0.05:
            theta_b = float(normalize(theta_a + pi))
        return theta_a, theta_b

    def gen_single_trial(self):
        raise NotImplementedError("gen_single_trial function must be implemented in subclass!")

    def draw(self, save=False, figname="task", n_batches=4):
        fig, axs = plt.subplots(5, n_batches, figsize=(8, 8), sharex=True, sharey=True)

        def round2(num):
            return str(round(num, 2))

        for b in range(n_batches):
            fix, (stim1, amp1), (stim2, amp2), resp, sacc = self.gen_single_trial()
            axs[0, b].plot(fix.squeeze(), color="k")
            axs[1, b].plot(stim1[:, 0], color="r")
            axs[1, b].plot(stim1[:, 1], color="r", linestyle="--")
            axs[2, b].plot(stim2[:, 0], color="b")
            axs[2, b].plot(stim2[:, 1], color="b", linestyle="--")
            axs[3, b].plot(resp.squeeze(), color="m")
            axs[4, b].plot(sacc.squeeze(), color="k")

            axs[0, b].set_title(f"Batch {b}")
            axs[1, b].set_title("A = " + round2(float(np.max(amp1))))
            axs[2, b].set_title("A = " + round2(float(np.max(amp2))))
            axs[3, b].set_title("θ = " + round2(float(np.max(np.abs(resp)))))

            for i in range(5):
                vis.set_invisible(axs[i, b])

        ylabels = ["Fixation", "Stimulus 1", "Stimulus 2", "Response", "Saccade"]
        for i in range(5):
            axs[i, 0].set_ylabel(ylabels[i])

        plt.tight_layout()
        if save:
            vis.savefig(os.path.join(SAVE_DIR, f"{figname}.png"), clear=True, close=True)


class _GoFamily(Task):
    """
    Go / Anti family.

    One stimulus direction is placed on a random modality. ``anti`` responds at θ+π.
    ``timing`` is ``"go"`` (stimulus stays on through the response), ``"rt"``
    (respond at onset), or ``"dly"`` (transient stimulus, delay, then response).
    """

    def __init__(self, hparams: dict, T: int = 200):
        super().__init__(T)
        self.hps = hparams

    def gen_single_trial(self):
        timing = self.hps["timing"]
        stim, amp = self.blank_modality()

        if timing == "rt":
            onset, dur = self.fit_epochs(
                np.random.randint(30, 50),
                np.random.randint(30, 50),
            )
            theta = self.sample_direction()
            self.write_pulse(stim, amp, 0, theta, 1.0, onset, onset + dur)
            go_time = onset
            fix = self.gen_fixation(self.T)
        elif timing == "go":
            onset, hold = self.fit_epochs(
                np.random.randint(20, 40),
                np.random.randint(40, 80),
            )
            go_time = onset + hold
            theta = self.sample_direction()
            # Stimulus has no offset when fixation is released.
            self.write_pulse(stim, amp, 0, theta, 1.0, onset, self.T)
            fix = self.gen_fixation(go_time)
        elif timing == "dly":
            onset, stim_dur, dly_dur = self.fit_epochs(
                np.random.randint(30, 50),
                np.random.randint(30, 50),
                np.random.randint(30, 50),
            )
            go_time = onset + stim_dur + dly_dur
            theta = self.sample_direction()
            self.write_pulse(stim, amp, 0, theta, 1.0, onset, onset + stim_dur)
            fix = self.gen_fixation(go_time)
        else:
            raise ValueError(f"Unknown timing: {timing}")

        ch1, ch2 = self.assign_modality(stim, amp)
        psi = theta + pi if self.hps["anti"] else theta
        resp = self.gen_response(psi, go_time)
        sacc = self.gen_fixation(go_time)
        return fix, ch1, ch2, resp, sacc


class Decision(Task):
    """
    Decision-making and delayed decision-making.

    Two directions are the two alternative slots. ``modalities`` is ``"1"``,
    ``"2"``, or ``"both"``. ``rule`` selects the stronger alternative from
    modality 1, modality 2, or the sum of both. ``offset`` presents the
    alternatives one after another instead of together.
    """

    def __init__(self, hparams: dict, T: int = 200):
        super().__init__(T)
        self.hps = hparams

    def _evidence(self, s1, s2):
        rule = self.hps["rule"]
        if rule == "mod1":
            return s1
        if rule == "mod2":
            return s2
        if rule == "sum":
            return s1 + s2
        raise ValueError(f"Unknown rule: {rule}")

    def _fill(self, stim, amp, strengths, theta_a, theta_b, spans):
        """``spans`` is ``((start_a, end_a), (start_b, end_b))``."""
        (a0, a1), (b0, b1) = spans
        self.write_pulse(stim, amp, 0, theta_a, strengths[0], a0, a1)
        self.write_pulse(stim, amp, 1, theta_b, strengths[1], b0, b1)

    def gen_single_trial(self):
        theta_a, theta_b = self.two_directions()
        use = self.hps["modalities"]
        s1 = np.zeros(N_ALTS)
        s2 = np.zeros(N_ALTS)
        if use == "both" and self.hps["rule"] == "sum":
            # Draw the combined evidence, then split it across modalities.
            combined = np.asarray(self.dm_amps(), dtype=float)
            frac = np.random.uniform(0.2, 0.8, size=N_ALTS)
            s1 = combined * frac
            s2 = combined * (1.0 - frac)
        else:
            if use in ("1", "both"):
                s1[:] = self.dm_amps()
            if use in ("2", "both"):
                s2[:] = self.dm_amps()

        if self.hps["offset"]:
            t1, d1, gap, d2, post = self.fit_epochs(
                np.random.randint(20, 40),
                np.random.randint(20, 35),
                np.random.randint(20, 40),
                np.random.randint(20, 35),
                np.random.randint(15, 30),
            )
            spans = ((t1, t1 + d1), (t1 + d1 + gap, t1 + d1 + gap + d2))
            go_time = spans[1][1] + post
        else:
            start, stim_dur = self.fit_epochs(
                np.random.randint(30, 50),
                np.random.randint(30, 50),
            )
            spans = ((start, start + stim_dur), (start, start + stim_dur))
            go_time = start + stim_dur

        stim1, amp1 = self.blank_modality()
        stim2, amp2 = self.blank_modality()
        if use in ("1", "both"):
            self._fill(stim1, amp1, s1, theta_a, theta_b, spans)
        if use in ("2", "both"):
            self._fill(stim2, amp2, s2, theta_a, theta_b, spans)

        evidence = self._evidence(s1, s2)
        choice = theta_a if evidence[0] > evidence[1] else theta_b
        fix = self.gen_fixation(go_time)
        resp = self.gen_response(choice, go_time)
        sacc = self.gen_fixation(go_time)
        return fix, (stim1, amp1), (stim2, amp2), resp, sacc


class Match(Task):
    """
    Delayed match / non-match.

    Two stimuli appear in sequence on one random modality, separated by a delay.
    A go trial reports the second direction and releases fixation. A no-go trial
    holds fixation and gives no directional response.

    ``cond`` is ``"point"`` (same direction) or ``"category"`` (same half-circle).
    ``nogo`` flips the rule (DNMS / DNMC).
    """

    def __init__(self, hparams: dict, T: int = 200):
        super().__init__(T)
        self.hps = hparams
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
            cats = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * pi
            theta1 = float(np.random.choice(cats))
            if np.random.randint(2):
                pool = [a for a in cats if self.same_category(theta1, a)]
            else:
                pool = [a for a in cats if not self.same_category(theta1, a)]
            theta2 = float(np.random.choice(pool))
        else:
            theta1 = self.sample_direction()
            if np.random.randint(2):
                theta2 = theta1
            else:
                theta2 = float(normalize(theta1 + pi + np.random.uniform(-pi / 6, pi / 6)))
                if abs(theta2) <= 0.05:
                    theta2 = float(normalize(theta1 + pi))

        stim, amp = self.blank_modality()
        self.write_pulse(stim, amp, 0, theta1, 1.0, t1, t1 + d1)
        self.write_pulse(stim, amp, 0, theta2, 1.0, t2, t2 + d2)
        ch1, ch2 = self.assign_modality(stim, amp)

        matched = bool(self.same(theta1, theta2))
        should_go = (not matched) if self.hps.get("nogo", False) else matched

        fix = self.gen_fixation(go_time)
        if should_go:
            sacc = self.gen_fixation(go_time)
            resp = self.gen_response(theta2, go_time)
        else:
            sacc = self.gen_fixation(self.T)
            resp = np.zeros((self.T, 1))
        return fix, ch1, ch2, resp, sacc

    @staticmethod
    def same_category(ang1, ang2):
        """Two broad categories: the two half-circles split at 0."""
        is_pos1 = normalize(ang1) > 0
        is_pos2 = normalize(ang2) > 0
        return not np.logical_xor(is_pos1, is_pos2)

    @staticmethod
    def same_value(ang1, ang2):
        return abs(normalize(ang1 - ang2)) < 1e-6


# Yang et al. 20-rule set.
task_map = {
    "fd_go": [_GoFamily, {"timing": "go", "anti": False}],
    "rt_go": [_GoFamily, {"timing": "rt", "anti": False}],
    "dly_go": [_GoFamily, {"timing": "dly", "anti": False}],
    "fd_go_anti": [_GoFamily, {"timing": "go", "anti": True}],
    "rt_go_anti": [_GoFamily, {"timing": "rt", "anti": True}],
    "dly_go_anti": [_GoFamily, {"timing": "dly", "anti": True}],
    # Simultaneous DM: alternatives live in one modality, or in both.
    "dm_1": [Decision, {"offset": False, "modalities": "1", "rule": "mod1"}],
    "dm_2": [Decision, {"offset": False, "modalities": "2", "rule": "mod2"}],
    "ctxt_dm_1": [Decision, {"offset": False, "modalities": "both", "rule": "mod1"}],
    "ctxt_dm_2": [Decision, {"offset": False, "modalities": "both", "rule": "mod2"}],
    "ctxt_dm_max": [Decision, {"offset": False, "modalities": "both", "rule": "sum"}],
    # Sequential (delayed) DM.
    "dly_dm_mod_1": [Decision, {"offset": True, "modalities": "1", "rule": "mod1"}],
    "dly_dm_mod_2": [Decision, {"offset": True, "modalities": "2", "rule": "mod2"}],
    "dly_dm_1": [Decision, {"offset": True, "modalities": "both", "rule": "mod1"}],
    "dly_dm_2": [Decision, {"offset": True, "modalities": "both", "rule": "mod2"}],
    "dly_dm_max": [Decision, {"offset": True, "modalities": "both", "rule": "sum"}],
    "dms": [Match, {"cond": "point", "nogo": False}],
    "dms_nogo": [Match, {"cond": "point", "nogo": True}],
    "dmc": [Match, {"cond": "category", "nogo": False}],
    "dmc_nogo": [Match, {"cond": "category", "nogo": True}],
}

YANG20_TASKS = (
    "fd_go",
    "rt_go",
    "dly_go",
    "fd_go_anti",
    "rt_go_anti",
    "dly_go_anti",
    "dm_1",
    "dm_2",
    "ctxt_dm_1",
    "ctxt_dm_2",
    "ctxt_dm_max",
    "dly_dm_mod_1",
    "dly_dm_mod_2",
    "dly_dm_1",
    "dly_dm_2",
    "dly_dm_max",
    "dms",
    "dms_nogo",
    "dmc",
    "dmc_nogo",
)
