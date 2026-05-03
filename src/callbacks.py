import math
import os
import h5py
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

import utils.visualization_utils as vis

from copy import deepcopy
from utils.common_utils import area_activity_to_poisson_counts, sigmoid, one_hot_encode


# Save directory
SAVE_DIR = "./graphs/"


class OnEpochStartCalls(pl.Callback):
    """
    Custom callbacks at the start of train/validation epochs.
    """
    def __init__(
        self,
        callbacks: list,
        priority: int = 1
    ):
        self.priority = priority
        self.callbacks = callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def run(self, trainer, pl_module, step_type):
        kwargs = {"step_type": step_type}
        for i, callback in enumerate(self.callbacks):
            # Use log if present as kwargs
            if callback.name == "log": kwargs["metrics"] = callback.metrics
                
            if step_type in callback.run_steps:
                callback.run(trainer, pl_module, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        self.run(trainer, pl_module, "train")
            
    def on_validation_epoch_start(self, trainer, pl_module):
        self.run(trainer, pl_module, "valid")


class OnEpochEndCalls(pl.Callback):
    """
    Custom callbacks at the end of train/validation epochs.
    """
    def __init__(
        self,
        callbacks: list,
        priority: int = 1
    ):
        self.priority = priority
        self.callbacks = callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def run(self, trainer, pl_module, step_type):
        kwargs = {"step_type": step_type}
        for i, callback in enumerate(self.callbacks):
            # Use log if present as kwargs
            if callback.name == "log": kwargs["metrics"] = callback.metrics

            if step_type in callback.run_steps:
                callback.run(trainer, pl_module, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "train")
            
    def on_validation_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "valid")
        

class Log:
    """
    Keeps a running history of train/validation metrics so other callbacks can plot or
    summarize how the run is progressing.
    """
    def __init__(
        self,
        run_steps: list = ["train", "valid"]
    ):
        self.name = "log"
        self.run_steps_count = 0
        self.run_steps = run_steps
        self.run_steps_copy = deepcopy(run_steps)
        self.metrics = {}
    
    def run(self, trainer, pl_module, **kwargs):
        if len(self.run_steps_copy) > 0:
            try:
                self.run_steps_copy.remove(kwargs["step_type"])
                for key in trainer.logged_metrics.keys():
                    self.metrics[key] = []
            except:
                pass
        
        for key, value in trainer.logged_metrics.items():
            self.metrics[key].append(value.item())
        

class SaveAsH5:
    """
    Periodically saves model tensors (hidden states, messages, ground truth, info, inputs)
    to data.h5 for further analysis and visualization.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        ground_truth: list = [],
        run_steps: list = ["valid"],
        poisson_activity: bool = False,
        poisson_dt: float = 0.1,
        poisson_rate_max: float = 50.0,
        poisson_seed: int | None = 42,
    ):
        """
        Args:
            log_every_n_epochs: The frequency of saving the data.
            ground_truth: The ground truth channels to save.
            run_steps: The steps to run the callback on.
            poisson_activity: Whether to transform area hidden state activity to Poisson counts in ``data.h5``.
            poisson_dt: The time bin in seconds.
            poisson_rate_max: The maximum rate (Hz).
            poisson_seed: RNG seed for Poisson draws (``None`` = nondeterministic each save).
        """
        self.name = "saveash5"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.ground_truth = ground_truth
        self.best = 1e10
        self.poisson_activity = poisson_activity
        self.poisson_dt = poisson_dt
        self.poisson_rate_max = poisson_rate_max
        self.poisson_seed = poisson_seed

    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
    
        print("Saving data from epoch ", trainer.current_epoch, "...")

        # Get task_reward info
        ground_truth_arr = pl_module.current_batch
        info = pl_module.current_info
        assert len(ground_truth_arr) == len(self.ground_truth)
        
        override = False
        new = kwargs["metrics"]["valid/loss"][-1]
        if new < self.best:
            self.best = new
            override = True
            print("OVERRIDE data.h5")
        
        if override:
            with h5py.File("data.h5", "w") as file:
                group = file.create_group("0") # session 0

                # Save hidden states
                for area_name, arr in pl_module.hidden_states.items():
                    data = arr.cpu().detach().numpy()
                    if self.poisson_activity:
                        data = area_activity_to_poisson_counts(data, self.poisson_dt, self.poisson_rate_max, seed=self.poisson_seed)
                    h5ds = group.create_dataset(f"area-{area_name}", data=data)
                    h5ds.attrs["type"] = "hidden_state"
                    if self.poisson_activity:
                        h5ds.attrs["representation"] = "poisson_counts"

                # Save messages
                for mes_name, arr in pl_module.save_var._asdict().items():
                    h5ds = group.create_dataset(f"message-{mes_name}", data=arr.cpu().detach().numpy())
                    h5ds.attrs["type"] = "message"

                # Save ground truth
                for ig in range(len(self.ground_truth)):
                    h5ds = group.create_dataset(f"truth-{self.ground_truth[ig]}", data=ground_truth_arr[ig].cpu().detach().numpy())
                    h5ds.attrs["type"] = "ground_truth"

                # Save info
                for info_name, info_val in info.items():
                    h5ds = group.create_dataset(f"info-{info_name}", data=info_val.cpu().detach().numpy())
                    h5ds.attrs["type"] = "info"
                    
                # Save inputs
                if hasattr(pl_module, "inputs"):
                    for area_name, arr in pl_module.inputs.items():
                        h5ds = group.create_dataset(f'inputs-{area_name}', data=arr.cpu().detach().numpy())
                        h5ds.attrs["type"] = "inputs"


class HistoryPlot:
    """
    Multi-panel figure per brain area: inputs vs messages over time, with latent traces
    across memory slots. Summarizes one batch element for qualitative analysis.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
    ):
        self.name = "historyplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.cutoff = 50
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
        if trainer.current_epoch <= 1: return

        # Get hparams
        hps = pl_module.hparams
        T = hps.memory
        num_rows, num_cols = hps.num_areas, 1 + T
        
        # Get data to plot
        inp = pl_module.current_batch[0].cpu().detach().numpy()
        mesgs = pl_module.save_var.mesgs.cpu().detach().numpy()
        latents = pl_module.save_var.latents.cpu().detach().numpy()
        batch_size, time_size = inp.shape[:2]

        # Plot
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*3),
            sharex = False,
            sharey = False,
        )
        vis.common_row_ylabel(fig, pl_module.area_names, (num_rows, num_cols))
        vis.common_label(fig, "time step", "")
        
        b = 0
        latents_reshape = latents.reshape(batch_size, time_size, hps.total_mesgs, hps.memory)
        
        colors = sns.color_palette("Set2", hps.total_ranks)
        for ia, (area_name, area) in enumerate(pl_module.areas.items()):
            
            ibase = hps.num_areas * ia
            isource = torch.arange(hps.num_areas) + ibase
            mesgs_ia = mesgs[b, :, pl_module.get_index_mesg(*isource)] # shape = (t, num mesgs)
            
            isource2 = []
            for ic, conn in enumerate(hps.connectome[ia]):
                if conn > 0:
                    isource2 += pl_module.get_index_rank(ic)
            inp_ia = inp[b, :, isource2] # shape = (t, num mesgs)
            
            for im, mesg in enumerate(mesgs_ia):
                axs[ia][0].plot(inp_ia[im][:self.cutoff] + im * 2, color=colors[im], alpha=0.5)
                axs[ia][0].plot(mesg[hps.lag:self.cutoff+hps.lag] + im * 2, color=colors[im], linestyle="--")

                for t in range(hps.memory):
                    latents_ia = latents_reshape[b, :, pl_module.get_index_mesg(*isource), t]
                    axs[ia][t+1].plot(inp_ia[im, t:-(T-t)][:self.cutoff] + im * 2, color=colors[im], alpha=0.5)
                    axs[ia][t+1].plot(latents_ia[im, hps.memory:][hps.lag*2:self.cutoff+hps.lag*2] + im * 2, color=colors[im], linestyle="--")
                    
        vis.savefig(f"HistorySummary_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True) 


class MesgSummaryPlot:
    """
    Grid: top row = source/target reconstruction losses over epochs; below = per-area
    input vs predicted message traces (source and target columns).
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
    ):
        self.name = "mesgsummaryplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
        if trainer.current_epoch <= 1: return

        metrics = kwargs["metrics"]
        hps = pl_module.hparams
        num_rows, num_cols = 1 + max([hps.num_source, hps.num_target]), 2
        
        # Get messages to plot
        inp = pl_module.current_batch[0].cpu().detach().numpy()
        s_to_t = pl_module.save_var.s_to_t.cpu().detach().numpy()
        outputs = pl_module.save_var.outputs.cpu().detach().numpy()

        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*3),
            sharex = False,
            sharey = False,
        )
        
        # Get color
        colors = sns.color_palette("Set2", max([hps.input_dim, 2]))
        
        # Plot losses
        axs[0][0].plot(metrics[f"train/mr_s_loss"][1:], label="train", color=colors[0])
        axs[0][0].plot(metrics[f"valid/mr_s_loss"][1:], label="valid", color=colors[1])
        axs[0][1].plot(metrics[f"train/mr_t_loss"][1:], label="train", color=colors[0])
        axs[0][1].plot(metrics[f"valid/mr_t_loss"][1:], label="valid", color=colors[1])
        axs[0][0].set_ylabel("source loss")
        axs[0][1].set_ylabel("target loss")
        
        # Plot messages
        b = 0
        for ia, (area_name, area) in enumerate(pl_module.source_areas.items()):
            for idim in range(hps.input_dim):
                axs[ia + 1][0].plot(inp[b, :-hps.lag, idim], color=colors[ia], alpha=0.5)
                axs[ia + 1][0].plot(s_to_t[b, ia, hps.lag:, idim], color=colors[ia], linestyle="--")
            axs[ia + 1][0].set_ylabel(area_name)
                
        for ia, (area_name, area) in enumerate(pl_module.target_areas.items()):
            mask = hps.mask[ia].reshape(1, 1, *hps.mask[ia].shape).numpy()
            inp_tile = np.tile(np.expand_dims(inp, 2), (1, 1, hps.num_source, 1))
            inp_weighted = np.sum( inp_tile * mask , axis=(2,3)) # shape = (batch, time)

            axs[ia + 1][1].plot(inp_weighted[b, :-2*hps.lag], color="k", alpha=0.5)
            axs[ia + 1][1].plot(outputs[b, ia, 2*hps.lag:, 0], color="k", linestyle="--")
            axs[ia + 1][1].set_ylabel(area_name)

        vis.savefig(f"MesgSummary_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)            


class ProctorSummaryPlot:
    """
    2x2 summary of logged loss and accuracy curves (train vs valid), driven by metric key names.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
        multiple_acc: bool = False,
    ):
        self.name = "proctorsummaryplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.multiple_acc = multiple_acc
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return

        metrics = kwargs["metrics"]
        num_rows, num_cols = 2, 2

        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*3),
            sharex = True,
            sharey = False,
        )
        vis.common_row_ylabel(fig, ["loss", "accuracy"], (num_rows, num_cols))
        
        i_train, i_valid = 0, 0
        colors = sns.color_palette("Set2", len(metrics))
        for key, value in metrics.items():
            step_type, name = key.split("/")
            if ('loss' not in name) and ('acc' not in name): continue # skip if not loss or acc
            
            is_loss = "loss" in name
            is_valid = step_type == "valid"
            idx = i_valid if is_valid else i_train
            axs[1-int(is_loss)][int(is_valid)].plot(value, label=name, color=colors[idx])
            axs[1-int(is_loss)][int(is_valid)].set_title(step_type)
            if is_loss or self.multiple_acc: axs[1-int(is_loss)][int(is_valid)].legend()

            if is_valid: i_valid += 1
            else: i_train += 1

        vis.savefig(f"ProctorSummary_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)


class MessagePlot:
    """
    Per-area, per-batch-column overlay of ground-truth channel vs predicted messages.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
        data_type = "normal",
    ):
        self.name = "messageplot2"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.data_type = data_type
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return

        num_rows, num_cols = pl_module.hparams.num_areas, 5
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*2),
            sharex = True,
        )
        vis.common_col_title(fig, [f"batch {i}" for i in range(num_cols)], (num_rows, num_cols))
        vis.common_row_ylabel(fig, pl_module.area_names, (num_rows, num_cols))
        vis.common_label(fig, "time step", "")

        # Plot
        for b in range(num_cols):
            for m in range(num_rows):
                
                if self.data_type == "circular":
                    true = self.normalize_angle(pl_module.current_batch[0][b, :, m])
                    pred = self.normalize_angle(pl_module.save_var.mesgs[b, :, m])
                else:
                    true = pl_module.current_batch[0][b, :, m]
                    pred = pl_module.save_var.mesgs[b, :, m]
                
                axs[m][b].plot(true, "k")
                axs[m][b].plot(pred, "b--")

        vis.savefig(f"Message_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)
    
    # Helper function to normalize angle of a channel
    def normalize_angle(self, angle):
        return np.angle(np.exp(1j * angle))
    

class PassDecisionPlot:
    """
    Compares cumulative input (decision integrator) to model-predicted decision traces.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"]
    ):
        self.name = "passdecisionplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return

        num_rows, num_cols = 2, 4
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*2),
            sharex = True,
        )
        vis.common_col_title(fig, [f"batch {i}" for i in range(num_cols)], (num_rows, num_cols))
        vis.common_row_ylabel(fig, ["dec 1", "dec 2"], (num_rows, num_cols))
        vis.common_label(fig, "time step", "")
        
        inp, latent, go, ctxt, action = pl_module.current_batch
        dec = torch.cumsum(inp, dim=1)

        # Plot true and predicted decision traces
        for b in range(num_cols):
            axs[0][b].plot(dec.cpu().detach().numpy()[b, :, 0], "k", label="true")
            axs[0][b].plot(pl_module.save_var.d.cpu().detach().numpy()[b, :, 0], "b", label="pred")
            axs[1][b].plot(dec.cpu().detach().numpy()[b, :, 1], "k", label="true")
            axs[1][b].plot(pl_module.save_var.d.cpu().detach().numpy()[b, :, 1], "b", label="pred")
            
        vis.savefig(f"PassDecision_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)
        

class HiddenUnitsPlot:
    """
    View a random subset of hidden unit time courses per area and batch column.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"]
    ):
        self.name = "hiddenunitsplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return

        # Get number of areas and number of rows and columns
        na = len(pl_module.areas)
        num_rows, num_cols = na, 4
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*2),
            sharex = True,
        )
        vis.common_col_title(fig, [f"batch {i}" for i in range(num_cols)], (num_rows, num_cols))
        vis.common_row_ylabel(fig, pl_module.area_names, (num_rows, num_cols))
        vis.common_label(fig, "time step", "")
        
        # Plot hidden unit time courses
        for ia, (area_name, hs) in enumerate(pl_module.hidden_states.items()):
            for b in range(num_cols):
                for nn in np.random.choice(hs.shape[-1], size=10).astype(int):
                    axs[ia, b].plot(hs[b, :, nn].cpu().detach().numpy())
                vis.set_invisible(axs[ia, b])
        
        vis.savefig(f"HiddenUnits_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)


class TaskRespPlot:
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
        num_batches: int = 4,
    ):
        """
        Plot task and response, where each column is a batch.
        Row 1 are the fixation target and saccade (after sigmoid).
        Row 2 are the response target (after argmax) and output (population).
        """
        self.name = "taskrespplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.num_batches = num_batches

    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if trainer.current_epoch <= 1:
            return

        hps = pl_module.hparams
        num_rows, num_cols = 2, max([self.num_batches, hps.num_tasks])

        # Get data to plot
        fix, stim1, amp1, stim2, amp2, task, resp, sacc = pl_module.current_batch
        resp_angles = resp.cpu().detach()

        # One-hot encode response
        resp = one_hot_encode(resp, [-math.pi, math.pi], hps.num_angles).cpu().detach()
        outputs = pl_module.outputs.cpu().detach()

        # Plot, each column is a batch
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols * 3, num_rows * 3),
            sharex=False,
            sharey=False,
        )

        # Get color, each area is a color
        colors = sns.color_palette("hls", hps.num_areas)
        for b in range(num_cols):
            axs[0, b].plot(sacc[b, :, 0].cpu().detach().numpy(), "k")
            for n in range(hps.num_areas):
                axs[0, b].plot(
                    sigmoid(pl_module.save_var.latents[b, :, n].cpu().detach().numpy()),
                    color=colors[n],
                    label=f"A{n}",
                )
            # Plot response class and mask
            resp_class = torch.argmax(resp, dim=2).numpy()
            resp_mask = torch.where(
                resp_angles != 0, torch.tensor(1), torch.tensor(0)
            ).numpy()[..., 0]
            angle_class = torch.argmax(outputs, dim=2).numpy()
            axs[1, b].plot(resp_class[b, :] * resp_mask[b, :], "k")
            axs[1, b].plot(
                angle_class[b, :] * resp_mask[b, :]
                + outputs[b, :].numpy().mean(axis=-1) * (1 - resp_mask[b, :]),
                color="b",
                linestyle="--",
            )

        axs[0, 0].legend()
        vis.common_col_title(fig, [f"Batch {i}" for i in range(num_cols)], axs.shape)
        vis.common_row_ylabel(fig, ["Fixation", "Response"], axs.shape)
        vis.savefig(
            f"TaskResp_epoch={trainer.current_epoch}.png",
            folders=[SAVE_DIR],
            close=True,
        )
