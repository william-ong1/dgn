import os
import h5py
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy

import dgn.utils.visualization_utils as vis

SAVE_DIR = "./graphs/"

class OnEpochEndCalls(pl.Callback):
    def __init__(self,
                 callbacks: list,
                 priority: int = 1):
        self.priority = priority
        self.callbacks = callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def run(self, trainer, pl_module, step_type):
        kwargs = {"step_type": step_type}
        for i, callback in enumerate(self.callbacks):
            # Use log if present as kwargs
            if callback.name == "log": kwargs["metrics"] = callback.metrics
            
            # run callback according to step_type
            if step_type in callback.run_steps:
                new_kwargs = callback.run(trainer, pl_module, **kwargs)
                
    def on_train_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "train")
            
    def on_validation_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "valid")
        
class OnEpochStartCalls(pl.Callback):
    def __init__(self,
                 callbacks: list,
                 priority: int = 1):
        self.priority = priority
        self.callbacks = callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def run(self, trainer, pl_module, step_type):
        kwargs = {"step_type": step_type}
        for i, callback in enumerate(self.callbacks):
            # Use log if present as kwargs
            if callback.name == "log": kwargs["metrics"] = callback.metrics
            
            # run callback according to step_type
            if step_type in callback.run_steps:
                new_kwargs = callback.run(trainer, pl_module, **kwargs)
                
    def on_train_epoch_start(self, trainer, pl_module):
        self.run(trainer, pl_module, "train")
            
    def on_validation_epoch_start(self, trainer, pl_module):
        self.run(trainer, pl_module, "valid")
            
class Log:
    def __init__(self,
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
        
class HistoryPlot:
    def __init__(self,
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

        metrics = kwargs["metrics"]
        hps = pl_module.hparams
        T = hps.memory
        num_rows, num_cols = hps.num_areas, 1 + T
        
        # Get messages to plot
        inp = pl_module.current_batch[0].cpu().detach().numpy()
        mesgs = pl_module.save_var.mesgs.cpu().detach().numpy()
        latents = pl_module.save_var.latents.cpu().detach().numpy()
        batch_size, time_size = inp.shape[:2]

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
    def __init__(self,
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
        b = 0 # Use first batch
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
    def __init__(self,
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
        
class ActionPlot:
    def __init__(self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"]
    ):
        self.name = "actionplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.sigmoid = lambda x: 1/(1+np.exp(-0.02 * x))
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
    
        # Get necessary info
        (inp, latent, go, ctxt, action) = pl_module.current_batch
        info = pl_module.current_info
        inp = inp.cpu().detach().numpy()
        latent = latent.cpu().detach().numpy()
        go = go.cpu().detach().numpy()
        ctxt = ctxt.cpu().detach().numpy()
        action = action.cpu().detach().numpy()
        dec = np.cumsum(inp, axis=1)
        a = pl_module.save_var.a.cpu().detach().numpy()
        d_to_m = pl_module.save_var.d_to_m.cpu().detach().numpy()
        h_m = pl_module.hidden_states["M"].cpu().detach().numpy()
        batch, time, _ = inp.shape
        hps = pl_module.hparams
        if not hps.lag:
            plag = 0
            nlag = None
        else:
            plag = hps.lag
            nlag = -hps.lag

        # Set up figure
        num_rows, num_cols = 2, 3
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*2),
        )

        # Plot action of 3 individual units
        b = 0 # Use the first batch only
        action_palette = sns.color_palette("dark:salmon_r", 3)
        for n in range(3):
            axs[1][0].plot(action[b, :, n], color=action_palette[n], alpha=0.5, linestyle="-")
            axs[1][0].plot(a[b, :, n], color=action_palette[n], alpha=1, linestyle="--")
            
        # Plot rotation in x-z and y-z plane
        for b in range(10):
            vis.color_time(axs[1][1], a[b, :, [0, 2]], cmap="coolwarm")
            vis.color_time(axs[1][2], a[b, :, [1, 2]], cmap="coolwarm")
            
        # Plot decision
        b = 0
        d_true_sign = np.sign(dec) # becomes -1, 0 (highly unlikely) or 1
        crpt_dec1 = np.expand_dims( np.linalg.norm(np.stack([dec[..., 0], latent[..., 0]], axis=2), axis=2), axis=-1 ) # shape = (batch, time, 1)
        crpt_dec2 = np.expand_dims( np.linalg.norm(np.stack([dec[..., 1], latent[..., 0]], axis=2), axis=2), axis=-1 ) # shape = (batch, time, 1)
        crpt_dec1 *= d_true_sign[..., :1]
        crpt_dec2 *= d_true_sign[..., 1:]
        crpt_dec = np.stack([crpt_dec1, crpt_dec2], axis=2)
        axs[0][0].plot(crpt_dec[b, :nlag, 0], color=action_palette[0], alpha=0.5, linestyle="-")
        axs[0][0].plot(crpt_dec[b, :nlag, 1], color=action_palette[1], alpha=0.5, linestyle="-")
        axs[0][0].plot(d_to_m[b, plag:, 0], color=action_palette[0], alpha=1, linestyle="--")
        axs[0][0].plot(d_to_m[b, plag:, 1], color=action_palette[1], alpha=1, linestyle="--")
        
        # Plot initial radius comparison
        r0_trues, r0_preds_x, r0_preds_y = [], [], []
        for b in range(batch):
            go_time = np.argmax(go[b, :, 0])
            choice = ctxt[b, go_time-10, 0].item()
            x0 = dec[b, go_time - plag, 0].item() if choice == 1 else dec[b, go_time - plag, 1].item()
            f0 = latent[b, go_time - plag, 0].item()
            r0_true = self.sigmoid(np.linalg.norm([x0, f0]))
            r0_trues.append(r0_true)
            
            r0_preds_x.append(abs(a[b, go_time, 0]))
            r0_preds_y.append(abs(a[b, go_time, 1]))
        axs[0][1].scatter(r0_trues, r0_preds_x, color="k", s=10)
        axs[0][2].scatter(r0_trues, r0_preds_y, color="k", s=10)
        axs[0][1].plot([0.5, 0.8], [0.5, 0.8], color="grey", linestyle="--")
        axs[0][2].plot([0.5, 0.8], [0.5, 0.8], color="grey", linestyle="--")

        vis.savefig(f"Action_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)
        
class MessagePlot:
    def __init__(self,
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
        
    def normalize_angle(self, angle):
        return np.angle(np.exp(1j * angle))
    
class PassDecisionPlot:
    def __init__(self,
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

        for b in range(num_cols):
            axs[0][b].plot(dec.cpu().detach().numpy()[b, :, 0], "k", label="true")
            axs[0][b].plot(pl_module.save_var.d.cpu().detach().numpy()[b, :, 0], "b", label="pred")
            axs[1][b].plot(dec.cpu().detach().numpy()[b, :, 1], "k", label="true")
            axs[1][b].plot(pl_module.save_var.d.cpu().detach().numpy()[b, :, 1], "b", label="pred")
            
        vis.savefig(f"PassDecision_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)
        
class HiddenUnitsPlot:
    def __init__(self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"]
    ):
        self.name = "hiddenunitsplot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return

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
        
        for ia, (area_name, hs) in enumerate(pl_module.hidden_states.items()):
            for b in range(num_cols):
                for nn in np.random.choice(hs.shape[-1], size=10).astype(int):
                    axs[ia, b].plot(hs[b, :, nn].cpu().detach().numpy())
                vis.set_invisible(axs[ia, b])
        
        vis.savefig(f"HiddenUnits_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)

class SaveAsH5:
    def __init__(self,
        log_every_n_epochs: int = 1,
        ground_truth: list = [],
        run_steps: list = ["valid"],
    ):
        self.name = "saveash5"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.ground_truth = ground_truth
        
        self.best = 1e10
        
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

                for area_name, arr in pl_module.hidden_states.items():
                    h5ds = group.create_dataset(f"area-{area_name}", data=arr.cpu().detach().numpy())
                    h5ds.attrs["type"] = "hidden_state"

                for mes_name, arr in pl_module.save_var._asdict().items():
                    h5ds = group.create_dataset(f"message-{mes_name}", data=arr.cpu().detach().numpy())
                    h5ds.attrs["type"] = "message"

                for ig in range(len(self.ground_truth)):
                    h5ds = group.create_dataset(f"truth-{self.ground_truth[ig]}", data=ground_truth_arr[ig].cpu().detach().numpy())
                    h5ds.attrs["type"] = "ground_truth"

                for info_name, info_val in info.items():
                    h5ds = group.create_dataset(f"info-{info_name}", data=info_val.cpu().detach().numpy())
                    h5ds.attrs["type"] = "info"
                    
                if hasattr(pl_module, "inputs"):
                    for area_name, arr in pl_module.inputs.items():
                        h5ds = group.create_dataset(f'inputs-{area_name}', data=arr.cpu().detach().numpy())
                        h5ds.attrs["type"] = "inputs"
                        
class get_default_conditions:
    def __init__(self, var_name="x0"):
        self.name = "default_condition"
        self.var_name = var_name
    
    def run(self, trainer, pl_module, **kwargs):
        info_dict = kwargs["info"][0]
        batch_size = info_dict[self.var_name].shape[0]
        indices = [np.arange(batch_size).astype(int)]
        categories = [0]
        
        pl_module.conditions = {0: (categories, indices)}
        