import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import networkx as nx

import utils.visualization_utils as vis

from utils.torch_utils import (
    RNNChannel,
    MLPBase,
    normalize_torch,
    one_hot_encode,
    sum_nested,
)

from utils.common_utils import (
    Messages,
    PassDecisionMotionMessages,
    get_insert_func,
    flatten
)


SAVE_DIR = "./graphs/"


class VariableNoise:
    """
    Variable noise update class for hidden state and channel noise for DGNs.
    """
    def __init__(self, init_noise, feature_dim, ema_decay=0.99, device='cuda', off=False):
        """
        Args:
            init_noise: Initial noise magnitude.
            feature_dim: Number of features.
            ema_decay: Exponential moving average decay rate.
            device: Device to use.
            off: If True, noise is turned off.
            noise_dist: Distribution of the noise.
            ema_norm: Exponential moving average of the noise.
        """
        self.init_noise = float(init_noise)
        self.feature_dim = feature_dim
        self.ema_decay = ema_decay
        self.device = device
        self.off = off

        self.noise_dist = torch.empty(feature_dim, device=device).uniform_(0.2, 1.0)
        self.ema_norm = torch.zeros(feature_dim, device=device)

    def __call__(self, data):
        B, T, F = data.shape

        # If noise is turned off:
        if self.off:
            return torch.full((F,), self.init_noise, device=data.device, dtype=data.dtype)

        # Compute per-feature norms
        with torch.no_grad():
            # Norm across time per sample, then average across batch
            mean_sq = data.pow(2).mean(dim=(0, 1))  # shape: (F,)
            norm = torch.sqrt(mean_sq + 1e-8)       # per-feature magnitude

            # Update EMA
            if self.ema_norm.sum() == 0: # first update → direct assign
                self.ema_norm = norm
            else:
                self.ema_norm = (self.ema_decay * self.ema_norm + (1 - self.ema_decay) * norm)

        scaled_noise = self.ema_norm * self.noise_dist
        return scaled_noise


class DGNBase(pl.LightningModule):
    """
    Base class for all Data Generating Networks (DGN) Lightning Modules.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function must be implemented in subclass!')
     
    def _shared_step(self, *args, **kwargs):
        raise NotImplementedError('_shared_step function must be implemented in subclass!')
        
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")
    
    def predict_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")
    
    def configure_optimizers(self):
        hps = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = hps.lr,
        )
        return optimizer

    def _scale_gru_input_weights_by_var(self) -> None:
        """
        Scale GRUCell/RNNCell input weights (weight_ih) to increase variance.
        """
        var_scale = float(getattr(self.hparams, "input_weight_init_var_scale", 1.0))
        if var_scale == 1.0:
            return
        std_scale = math.sqrt(var_scale)

        # Scale input weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.GRUCell, nn.RNNCell)):
                    m.weight_ih.mul_(std_scale)


class MemoryNetwork(DGNBase):
    """
    Multi-area recurrent memory network with a defined connectome for inter-area communication (DGN).
    """
    def __init__(
        self,
        ranks: list,
        connectome: list,
        
        lag: int = 1,
        memory: int = 5,
        noise: float = 0.0,
        channel_noise: float = 0.0,
        noise_type: str = 'fixed',
        channel_noise_type: str = 'fixed',
        
        hidden_size: int = 64,
        lr: float = 4.0e-3,
        input_weight_init_var_scale: float = 1.0,
        
        ext_input_dim: int = 0,
        ext_input_amp: int = -1,
        ext_input_perc: float = 0.0,
    ):
        """
        Args:
            ranks: Dimensionality of the private input for each area.
            connectome: Binary connectivity matrix C where C[i, j] ∈ {0, 1}
                indicates whether area j projects to area i. Diagonal entries
                must be 1, even though there is no explicit self-connection.
                
            lag: Time-step lag between inter-area communication.
            memory: Number of time steps each area must encode in its hidden
                unit activity.
            noise: Dynamic noise applied to hidden unit activity.
            channel_noise: Noise applied to inter-area communication signals.
            noise_type: Either 'fixed' or 'variable', specifying whether
                dynamic noise scales with the magnitude (standard deviation)
                of hidden unit activity.
            channel_noise_type: Same as `noise_type`, but applied to channel
                noise.

            hidden_size: Number of hidden units per area.
            lr: Learning rate.
            input_weight_init_var_scale: Scale factor for the initial variance
                of the RNN input weights.

            ext_input_dim: Dimensionality of external perturbation inputs.
                Perturbations are implemented as a step input from time steps
                100 to 120.
            ext_input_amp: Amplitude of external perturbation inputs.
            ext_input_perc: Fraction of trials in which one area receives a
                perturbation input. Trials are selected randomly according to
                this probability.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Setup hyperparameters
        hps = self.hparams
        hps.num_areas = len(ranks)
        hps.ranks = torch.Tensor(hps.ranks).to(int)
        hps.connectome = torch.Tensor(hps.connectome).to(int)
        
        # Setup external inputs if applicable
        if isinstance(hps.ext_input_dim, int): hps.ext_input_dim = [hps.ext_input_dim] * hps.num_areas
        
        # Build effectome: combines rank with connectome
        hps.effectome = torch.tile(hps.ranks.reshape(1, -1), (len(hps.ranks), 1)) * hps.connectome
        hps.mesgs_idx = hps.effectome.flatten()
        hps.total_mesgs = sum(hps.mesgs_idx) # input + communication dimensions total
        hps.total_ranks = sum(hps.ranks) # total dimensionality of all inputs
        
        # Build areas
        self.areas = nn.ModuleDict()
        for na in range(hps.num_areas):
            output_dims = hps.effectome[na]
            output_dims = [hps.ranks[na], sum(output_dims) * hps.memory]
            self.areas[f"A{na}"] = RNNChannel(
                sum(hps.effectome[na]) + hps.ext_input_dim[na],
                hps.hidden_size,
                output_dims,
                None,
            )
        self.area_names = list(self.areas.keys())
        
        # Loss functions and decoders
        self.mseloss = nn.MSELoss()
        
        # Areas are required to readout input/messages and 'memory' of input/messages
        # All input/messages are stored in `mesgs`
        # All memories are stored in `latents`
        # For convenience of accessing the slicing indices, the index functions
        # below returns the indices corresponding to each area
        def get_index_func(arr, scale=1):
            def inner(*idxs):
                res = []
                for idx in idxs:
                    base = sum(arr[:idx]) * scale
                    rank = arr[idx] * scale
                    res += list(range(base, base+rank))
                return res
            return inner
        
        # Indexing functions for mesgs, ranks, and latents
        self.get_index_mesg = get_index_func(hps.mesgs_idx)
        self.get_index_rank = get_index_func(hps.ranks)
        self.get_index_latent = get_index_func(hps.mesgs_idx, scale=hps.memory)
        
        # Build variable noise update class (turned off for 'fixed' noise type)
        assert self.hparams.noise_type in ['fixed', 'variable']
        self.noise_weight_generator = VariableNoise(
            self.hparams.noise,
            hps.num_areas * hps.hidden_size,
            device = 'cuda',
            off = (self.hparams.noise_type == 'fixed'),
        )

        # Build variable noise update class for channel noise
        assert self.hparams.channel_noise_type in ['fixed', 'variable']
        self.cnoise_weight_generator = VariableNoise(
            self.hparams.channel_noise,
            hps.total_mesgs,
            device = 'cuda',
            off = (self.hparams.channel_noise_type == 'fixed'),
        )

        # Initialize noise weights
        self.h_noise_weight = torch.ones(hps.num_areas * hps.hidden_size).to(self.device) * hps.noise
        self.c_noise_weight = torch.ones(hps.total_mesgs).to(self.device) * hps.channel_noise

        # Increase variance of the RNN input weights
        self._scale_gru_input_weights_by_var()
        
    def forward(self, inp, step_type):
        hps = self.hparams
        batch, time, _ = inp.shape # (batch, time, total rank)
        
        # Build storage
        self._build_save_var(batch, time) # build storage for mesgs, latents
        h = torch.zeros(hps.num_areas, batch, hps.hidden_size).to(self.device)
        mesgs = torch.zeros(batch, hps.total_mesgs).to(self.device) # (batch, total mesg channels)
        
        # Setup external input
        self.ext_inputs = {}
        for ia, area_name in enumerate(self.area_names):
            self.ext_inputs[area_name] = torch.zeros(batch, time, hps.ext_input_dim[ia], device=self.device)
            has_ext_inp = torch.bernoulli(torch.full((batch,), hps.ext_input_perc)) * hps.ext_input_amp # shape=(batch,)
            self.ext_inputs[area_name][:, 100:120] = torch.tile(has_ext_inp.reshape(-1, 1, 1), (1, 20, 1)).to(self.device)
        
        # Main loop
        for t in range(time):

            # Add hidden state noise
            h = h + torch.randn_like(h) * self.h_noise_weight.reshape(-1, 1, hps.hidden_size).to(self.device)
            
            # Setup variable storage per time t
            h_ias = []
            latents = torch.zeros(batch, hps.total_mesgs * hps.memory).to(self.device)
            mesgs_new = torch.zeros(batch, hps.total_mesgs).to(self.device)
            
            # Replace self-messages by input 
            for ia in range(hps.num_areas):
                idx = ia * hps.num_areas + ia
                mesgs[..., self.get_index_mesg(idx)] = inp[:, t, self.get_index_rank(ia)].to(self.device)
            
            # Forward pass through individual areas
            for ia, (area_name, area) in enumerate(self.areas.items()):
                
                # Indexing for inter-area communication
                ibase = hps.num_areas * ia
                isource = torch.arange(hps.num_areas) + ibase
                
                # Forward pass through individual area
                mesgs_inp = mesgs[..., self.get_index_mesg(*isource)]
                channel_noise = torch.randn_like(mesgs_inp) * self.c_noise_weight[self.get_index_mesg(*isource)].reshape(1, -1).to(self.device)
                mesgs_inp = mesgs_inp + channel_noise
                mesgs_inp = torch.cat([mesgs_inp, self.ext_inputs[area_name][:, t]], dim=-1).to(self.device) # add ext inp
                h_ia, mesgs_split = area(mesgs_inp, h[ia])
                
                # Insert into mesgs_new
                for ic in range(hps.num_areas):
                    isrc = hps.num_areas * ic + ia
                    mesg_idx = self.get_index_mesg(isrc)
                    if len(mesg_idx) > 0:
                        mesgs_new[..., mesg_idx] = mesgs_split[0].to(self.device)
                    
                # Save h_ia, latents
                h_ias.append(h_ia.unsqueeze(0))
                latents[..., self.get_index_latent(*isource)] = mesgs_split[-1].to(self.device)
            
            # Set new h, mesgs
            h = torch.cat(h_ias, dim=0) # shape = (num_areas, batch, hidden_dim)
            mesgs = mesgs_new
            
            # Save mesgs and hidden states
            self.save_var.mesgs[:, t] = mesgs # shape = (batch, t, total_mesgs)
            self.save_var.latents[:, t] = latents # shape = (batch, t, total_mesgs * memory)
            
            for ia, area_name in enumerate(self.areas):
                self.hidden_states[area_name][:, t] = h[ia]
                
        # Adjust noise weights
        self.h_noise_weight = self.noise_weight_generator(torch.cat([hs for hs in self.hidden_states.values()], dim=2))
        self.c_noise_weight = self.cnoise_weight_generator(self.save_var.mesgs)
                
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        inp, info = batch
        self.current_batch, self.current_info = inp, info
        
        # Forward pass through the model
        inp = inp[0]
        batch_size, time_size = inp.shape[:2]
        self.forward(inp, step_type)
        
        # Expose per-area ext-input tensors for export
        self.inputs = self.ext_inputs
        
        # Mesgs must reflect latent states of the source area
        inp_extended = []
        for ia in range(hps.num_areas):
            for ib in range(hps.num_areas):
                if hps.connectome[ia, ib]:
                    inp_extended.append(inp[..., self.get_index_rank(ib)])

        # Concatenate the input dimensions for each area
        inp_extended = torch.cat(inp_extended, dim=2)
        loss_mesg = self.mseloss(self.save_var.mesgs[:, hps.lag:], inp_extended[:, :-hps.lag]) # shape = (batch, time, total_mesgs)
        loss_mesg = torch.sum(loss_mesg) / batch_size
        
        # Decoder info must reflect the history of the latents
        # 4 nested for-loops:
        #     ia: target area
        #     ib: source area
        #     ic: a dimension of the input/memory
        #     ti: a time lag to store
        inp_hist = []
        for ia in range(hps.num_areas):
            for ib in range(hps.num_areas):
                
                # Only run if connection from area ib --> ia exists
                if hps.connectome[ia, ib]:
                    ics = self.get_index_rank(ib)
                    
                    # For each dimension of the input (as specified in `rank`)
                    for ic in ics:
                        
                        # For each time lag up to `memory`
                        for ti in range(hps.memory):
                            inp_hist.append(inp[:, ti: -(hps.memory - ti), ic].unsqueeze(2))
        
        inp_hist = torch.cat(inp_hist, dim=-1) # shape = (batch, time, total_mesgs * memory)

        # Compute loss for the history of the latents
        loss_hist = self.mseloss(self.save_var.latents[:, hps.memory + hps.lag*2:], inp_hist[:, :-hps.lag*2])
        loss_hist = torch.sum(loss_hist) / batch_size
        loss = loss_mesg + loss_hist
            
        # Log metrics
        metrics = {
            f"{step_type}/loss": loss,
            f"{step_type}/loss_mesg": loss_mesg,
            f"{step_type}/loss_hist": loss_hist,
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss
    
    def _build_save_var(self, batch_size, time):
        hps = self.hparams
        
        # Storage for mesgs, latents
        self.save_var = Messages(
            mesgs = torch.zeros(batch_size, time, hps.total_mesgs).to(self.device),
            latents = torch.zeros(batch_size, time, hps.total_mesgs * hps.memory).to(self.device),
        )
        
        # Storage for hidden units activity
        self.hidden_states = {}
        for area_name in self.areas:
            self.hidden_states[area_name] = torch.zeros(batch_size, time, hps.hidden_size).to(self.device)
            

class PassDecision(DGNBase):
    """
    Two-area recurrent pass-decision network with area-specific input signals and integration of task signals (DGN).
    """
    def __init__(
        self,
        lag: int = 0,

        noise_p: float = 0.0,
        noise_d: float = 0.0,

        hidden_size: int = 64,
        lr: float = 4.0e-3,
        input_weight_init_var_scale: float = 1.0,

        p_to_d_coef: float = 1.0,
        rep_coef: float = 0.0,
        binary_output: bool = True,
        rnn_nonlinearity: str = "tanh",
    ):
        """
        Args:
            lag: Time-step lag for pass-decision communication.

            noise_p: Dynamic noise applied to Pass-area hidden states.
            noise_d: Dynamic noise applied to Decision-area hidden states.

            hidden_size: Number of hidden units per area.
            lr: Learning rate.
            input_weight_init_var_scale: Scale factor for the initial variance
                of the RNN input weights.

            p_to_d_coef: Loss weight for matching the Pass-Decision channel to
                the raw input trajectory.
            rep_coef: Loss weight for matching Decision hidden readouts to the
                input (representation regularization).
            binary_output: If True, decision decoder uses sigmoid + BCE against
                a binarized cumulative input; if False, MSE against the
                cumulative input.
            rnn_nonlinearity: Nonlinearity for ``RNNChannel`` cells.
        """
        super().__init__()
        self.save_hyperparameters()
        hps = self.hparams

        # Fixed hps
        hps.input_dim = 2
        hps.channel_size = 2
        hps.output_size = hps.hidden_size

        # Pass area: stimulus + latent (input_dim)
        self.P_area = RNNChannel(hps.input_dim, hps.hidden_size, [hps.input_dim], 
                                 None, rnn_nonlinearity=hps.rnn_nonlinearity)
        # Decision area: input + latent (input_dim)
        self.D_area = RNNChannel(hps.input_dim, hps.hidden_size, [hps.channel_size], 
                                 None, rnn_nonlinearity=hps.rnn_nonlinearity)

        # Loss functions and decoders
        nonlinearity = "sigmoid" if hps.binary_output else None
        self.decoder = MLPBase([[hps.channel_size, hps.input_dim, nonlinearity]])
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCELoss() # expects probability (sigmoid)
        self.celoss = nn.CrossEntropyLoss() # does not expect probability (no sigmoid)
        
        # Decoder on Decision hidden state (representation matching to input)
        self.D_decoder = MLPBase([[hps.hidden_size, hps.input_dim, None]])

        # Increase variance of the RNN input weights
        self._scale_gru_input_weights_by_var()
        
    def forward(self, inp, latent, go, ctxt, step_type):
        hps = self.hparams
        batch, time, _ = inp.shape # (batch, time, input_dim)
        self._build_save_var(batch, time)

        # Initialize hidden states
        h_p = torch.zeros(batch, hps.hidden_size).to(self.device)
        h_d = torch.zeros(batch, hps.hidden_size).to(self.device)
        
        # Main loop
        for t in range(time):

            # Add hidden state noise
            h_p = h_p + torch.randn_like(h_p) * hps.noise_p
            h_d = h_d + torch.randn_like(h_d) * hps.noise_d
            
            # Forward pass through individual areas
            h_p, p_to_d = self.P_area(inp[:,t,:], h_p)
            h_d, d_to_m = self.D_area(p_to_d, h_d)
            
            # Decode decision
            d = self.decoder(d_to_m)
            d_rep = self.D_decoder(h_d)

            # Save variables
            self.save_var.p_to_d[:,t] = p_to_d
            self.save_var.d_to_m[:,t] = d_to_m
            self.save_var.d[:, t] = d
            self.save_var.d_rep[:, t] = d_rep
            
            # Save hidden states
            self.hidden_states["P"][:,t] = h_p
            self.hidden_states["D"][:,t] = h_d

        return list(self.save_var._asdict().values()) + list(self.hidden_states.values())
    
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        self.current_batch, self.current_info = batch

        # Setup lag
        if not hps.lag:
            plag = nlag = None
        else:
            plag = hps.lag
            nlag = -hps.lag
        
        # Forward pass
        inp, latent, go, ctxt, action = self.current_batch
        p_to_d, d_to_m, d_pred, a, d_rep, m_rep, h_p, h_d = self.forward(inp, latent, go, ctxt, step_type)
        
        # Get decision labels
        batch_size, time, _ = inp.shape
        dec = torch.cumsum(inp, dim=1) # shape = (batch, time, input_dim) still
        d_true_sign = torch.sign(dec) # becomes -1, 0 (highly unlikely) or 1
        d_true = (d_true_sign + 1) / 2 # becomes 0, 1
        
        # Loss for Pass on input
        p_to_d_loss = self.mseloss(p_to_d[:, plag:, :hps.input_dim], inp[:, :nlag])
            
        # Get decision decoder loss
        if hps.binary_output:
            d_loss = self.bceloss(d_pred[:, plag:].reshape(-1), d_true[:, :nlag].reshape(-1))
        else: # decode continuous, linear output instead
            d_loss = self.mseloss(d_pred[:, plag:].reshape(-1), dec[:, :nlag].reshape(-1))
        
        # Get area D representation loss
        d_rep_loss = self.mseloss(d_rep[:, plag:], inp[:, :nlag])
          
        # Get total loss
        loss = d_loss \
                + p_to_d_loss * hps.p_to_d_coef \
                + d_rep_loss * hps.rep_coef
        
        # Get accuracy
        d_label = (d_pred >= 0.5).float()
        accuracy = (d_true[:, :nlag] == d_label[:, plag:]).float().mean()

        # Log metrics
        metrics = {
            f"{step_type}/loss": loss,
            f"{step_type}/d_loss": d_loss,
            f"{step_type}/d_rep_loss": d_rep_loss * hps.rep_coef,
            f"{step_type}/p_to_d_loss": p_to_d_loss * hps.p_to_d_coef,
            f"{step_type}/accuracy": accuracy,
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss
    
    def _build_save_var(self, batch_size, time):
        hps = self.hparams

        # Storage for pass-decision communication
        self.save_var = PassDecisionMotionMessages(
            p_to_d = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            d_to_m = torch.zeros(batch_size, time, hps.channel_size).to(self.device),
            d = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            a = torch.zeros(0).to(self.device),
            d_rep = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            m_rep = torch.zeros(0).to(self.device),
        )

        # Storage for hidden states
        self.hidden_states = {
            "P": torch.zeros(batch_size, time, hps.output_size).to(self.device),
            "D": torch.zeros(batch_size, time, hps.output_size).to(self.device),
        }


class MultiTaskNet(DGNBase):
    """
    Multi-area recurrent network that performs multiple tasks jointly (DGN).
    """
    def __init__(
        self,
        num_areas: int,
        task_names: list,
        diagram: list,

        stim_input_areas: list = None,
        sacc_output_areas: list = None,
        sacc_scale: float = 1.0,
        
        delay: int = 0,
        hidden_size: int = 32,
        lr: float = 4.0e-3,
        input_weight_init_var_scale: float = 1.0,
        num_angles: int = 36,
        num_channels: int = 4,

        noise: float = 0.0,
        channel_noise: float = 0.0,
        noise_type: str = 'fixed',
        channel_noise_type: str = 'fixed',

        angle_start_epoch: int = 50,
        angle_increase_epoch: int = 100,
        angle_scale: float = 0.0,
        l1_start_epoch: int = 150,
        l1_increase_epoch: int = 150,
        l1_scale: float = 0.0,
        
    ):
        """
        Args:
            num_areas: Number of recurrent areas before the readout stage.
            task_names: Task names matching the datamodule; list order is the task index.
            diagram: Weighted edges (source, target, task_weight) for the task routing graph.

            stim_input_areas: Which area receives fix, stim1, stim2 (and optionally task); default is all A0.
            sacc_output_areas: Areas included in the saccade mask; unset becomes empty (all areas treated the same).
            sacc_scale: Weight on fixation / saccade loss.

            delay: Delay between messages between areas.
            hidden_size: Hidden units per area.
            lr: Learning rate for AdamW.
            input_weight_init_var_scale: Scale factor for the initial variance of the RNN input weights.
            num_angles: Bins for direction readout loss.
            num_channels: Size of each inter-area message.

            noise: Hidden-state noise scale.
            channel_noise: Inter-area message noise scale.
            noise_type: fixed or variable hidden noise.
            channel_noise_type: fixed or variable channel noise.

            angle_start_epoch: Start epoch for ramping in angle MSE.
            angle_increase_epoch: Ramp length for angle loss.
            angle_scale: Angle loss scale after ramp.
            l1_start_epoch: Start epoch for L1 ramp on communication weights.
            l1_increase_epoch: Ramp length for L1.
            l1_scale: L1 strength.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["task_names"])

        hps = self.hparams
        hps.task_names = task_names
        hps.num_tasks = len(hps.task_names)
        
        if not diagram:
            raise ValueError("MultiTaskNet requires a non-empty diagram (weighted edges list).")

        # Set default stimulus input areas
        if not stim_input_areas:
            hps.stim_input_areas = ["A0"] * 3
        else:
            hps.stim_input_areas = stim_input_areas

        # Set default saccade output areas
        if hps.sacc_output_areas is None:
            hps.sacc_output_areas = []

        # Build areas, insert function, slice function, loss functions
        self._build_areas()
        self.insert_func, self.slice_func = self.get_insert_func_nested(hps.total_mesgs)
        self.mseloss = nn.MSELoss(reduction="none")
        self.bceloss = nn.BCELoss(reduction="none")
        self.celoss = nn.CrossEntropyLoss(reduction="none")
        
        # Get indices corresponding to the input of an area, key of input_indices is the area_name,
        # values of input_indices is a list of tuples, (source_area, index of area_name in source_area's outputs)
        nested_idxs = {area_name: [] for area_name in self.area_names + [f"A{hps.num_areas}"]}
        for ia, area_name in enumerate(self.area_names + [f"A{hps.num_areas}"]):
            src_idxs = [int(node[1:]) for node in self.G.predecessors(area_name)]
            for src_idx in src_idxs:
                decs = sorted([int(node[1:]) for node in self.G.successors(f"A{src_idx}")])
                nested_idxs[area_name].append((src_idx, decs.index(ia)))
        self.input_indices = nested_idxs
        
        # Build mask, key of edges is the source area, value of edges is a list of target areas
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

        # Build channel noise update class
        assert self.hparams.channel_noise_type in ['fixed', 'variable']
        self.cnoise_weight_generator = VariableNoise(
            self.hparams.channel_noise,
            sum_nested(hps.total_mesgs),
            device = 'cuda',
            off = (self.hparams.channel_noise_type == 'fixed'),
        )

        # Initialize noise weights
        self.h_noise_weight = torch.ones(hps.hidden_size, hps.num_areas).to(self.device) * hps.noise
        self.c_noise_weight = torch.ones(sum_nested(hps.total_mesgs)).to(self.device) * hps.channel_noise
        
        # Increase variance of the RNN input weights
        self._scale_gru_input_weights_by_var()
        
    def forward(self, inp, step_type):
        hps = self.hparams
        batch, time, _ = inp[0].shape

        # Build save variables
        self._build_save_var(batch, time)
        
        # Initialize hidden states and messages
        h = torch.zeros(batch, hps.hidden_size * hps.num_areas).to(self.device)
        mesgs = torch.zeros(batch, sum_nested(hps.total_mesgs)).to(self.device)
        
        # Find task corresponding to each batch, shape = (batch,)
        task_idx = inp[3][:, 0, 0]
        
        # Initialize outputs
        self.outputs = []

        # Main loop
        for t in range(time):
            
            # Add noise to hidden states, h is shaped in the order of (hs of area 1, hs of area 2,...)
            h = h + torch.randn_like(h) * self.h_noise_weight.reshape(1, -1).to(self.device)
            
            # Setup variable storage per time t
            h_ias, fixs = [], []
            mesgs_new = torch.zeros(batch, sum_nested(hps.total_mesgs)).to(self.device)
            
            # Forward pass through each area
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

                # Forward pass through area
                h_ia, mesg_ias = area(inp_ia, h[:, ia*hps.hidden_size: (ia+1)*hps.hidden_size])
                h_ias.append(h_ia)
                
                # Store fixation 
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
                    
            # Gather input for output area
            inp_ia, mask = [], []
            for idx in self.input_indices[f"A{hps.num_areas}"]:
                
                # mask the input from each area according to whether their designated task matched current task
                mask = (task_idx.unsqueeze(1) == torch.Tensor(self.edges[idx[0]]).to(self.device)).any(dim=1).to(int)
                output = self.slice_func(mesgs, *idx)
                inp_ia.append(output * mask.reshape(-1, 1))
                
            # Forward pass through output area
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
            
            # Add noise to messages
            channel_noise = torch.randn_like(mesgs) * self.c_noise_weight.reshape(1, -1).to(self.device)
            mesgs = mesgs + channel_noise
            
        # Adjust noise weights
        self.h_noise_weight = self.noise_weight_generator(torch.cat([hs for hs in self.hidden_states.values()], dim=2)) # hidden states from all areas
        self.c_noise_weight = self.cnoise_weight_generator(self.save_var.mesgs)

        # Concatenate outputs
        self.outputs = torch.cat(self.outputs, dim=1)
        return self.outputs
            
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        inp, info = batch
        self.current_batch, self.current_info = inp, info
        
        # One-hot encode and forward pass
        fix, stim1, amp1, stim2, amp2, task, resp, sacc = inp
        resp = resp.float()
        polar1 = torch.cat([stim1, torch.tile(amp1.unsqueeze(1), (1, stim1.shape[1], 1))], dim=2)
        polar2 = torch.cat([stim2, torch.tile(amp2.unsqueeze(1), (1, stim2.shape[1], 1))], dim=2)
        self.forward([fix, polar1, polar2, task], step_type)
        
        # Fixation loss
        loss_sacc = self.bceloss(torch.sigmoid(self.save_var.latents), torch.tile(sacc, (1, 1, hps.num_areas)).float())
        loss_sacc = torch.mean(loss_sacc) * hps.sacc_scale
        
        # Response loss, compute ramp
        angle_ramp = self._compute_ramp(hps.angle_start_epoch, hps.angle_increase_epoch)
        
        # Get cosine, sine values of predicted and true responses
        cosine_pred = torch.cos(self.projs)
        sine_pred = torch.sin(self.projs)
        cosine_true = torch.cos(resp).to(self.device)
        sine_true = torch.sin(resp).to(self.device)
        
        # One hot encode target response
        resp_onehot = one_hot_encode(
            normalize_torch(resp), [-math.pi, math.pi], hps.num_angles
        )
        
        # Get resp_mask (assumes response angle is not exactly zero)
        resp_mask = torch.where(resp != 0, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) # shape = (batch, time, 1)
        
        # Get angle (mse) loss
        loss_angle = self.mseloss(cosine_true, cosine_pred) + self.mseloss(sine_true, sine_pred)
        loss_angle = torch.mean(loss_angle * resp_mask) * hps.angle_scale
        
        # Get response (cross entropy) loss
        loss_resp = self.celoss(flatten(self.outputs), flatten(resp_onehot)).reshape(*resp.shape)
        loss_resp = torch.mean(loss_resp * resp_mask)
        
        # Get response loss (mse) outside response period
        loss_base = self.mseloss(self.outputs, torch.zeros_like(self.outputs).to(self.device))
        base_mask = torch.where(resp == 0, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device))
        loss_base = torch.mean(loss_base * base_mask)
        
        # Regularization loss, compute ramp
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
            
        # Calculate accuracy (only using the last time point)
        pred   = torch.argmax(self.outputs[:, -1].detach().cpu(), dim=-1)   # (batch,)
        target = torch.argmax(resp_onehot[:, -1].detach().cpu(), dim=-1)    # (batch,)
        acc    = (pred == target).float()                                   # (batch,)

        per_task_acc = {}   # only tasks actually present in the batch

        # Get task indices for the last time point
        task_last = task[:, -1, 0].detach().cpu().long()   # (batch,)

        for itask, task_name in enumerate(hps.task_names):
            # Get boolean mask for samples of this task in the batch (1D, matches acc)
            mask = (task_last == itask)                    # (batch,)

            if mask.any():                                 # Skip if no samples of this task
                task_acc = acc[mask].mean().item()
                per_task_acc[task_name] = task_acc
        
        # Get total loss
        loss = loss_sacc + loss_resp + loss_base + loss_angle * angle_ramp + loss_l1 * l1_ramp
        metrics = {
            f"{step_type}/loss": loss,
            f"{step_type}/loss_sacc": loss_sacc,
            f"{step_type}/loss_resp": loss_resp,
            f"{step_type}/loss_base": loss_base,
            f"{step_type}/loss_angle": loss_angle,
            f"{step_type}/loss_l1": loss_l1,
        }
        
        # Add per-task accuracy
        for task_name, acc_val in per_task_acc.items():
            metrics[f"{step_type}/acc_{task_name}"] = acc_val
        
        # Log metrics
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=inp[0].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            try:
                _, ax = plt.subplots(1, 1)
                self.draw(ax)
                vis.savefig(f"{SAVE_DIR}network.png")
            except Exception:
                pass
        return self._shared_step(batch, "valid")

    def _build_specified_graph(self):
        """
        Build the task routing graph from the diagram.
        """
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
        """
        Build the areas of the network.
        """
        hps = self.hparams
        self.areas = nn.ModuleDict()
        
        G = self._build_specified_graph()

        # Handling output area
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
        
        # Initialize hidden states and inputs, projs for readout
        self.hidden_states = {}
        self.inputs = {}
        
        for ia, area_name in enumerate(self.area_names):
            self.hidden_states[area_name] = torch.zeros(batch_size, time, hps.hidden_size).to(self.device)

            # Get number of inputs, input for each area is the sum of the number of inputs of the source areas
            num_input = 0
            if area_name not in hps.stim_input_areas: pass
            else:
                if hps.stim_input_areas[0] == area_name: num_input += 1 # fix
                if hps.stim_input_areas[1] == area_name: num_input += 2 # stim1
                if hps.stim_input_areas[2] == area_name: num_input += 2 # stim2
                if hps.stim_input_areas[3] == area_name: num_input += 1 # task
            self.inputs[area_name] = torch.zeros(batch_size, time, num_input).to(self.device)
            
        # Initialize projs for readout
        self.projs = torch.zeros(batch_size, time, 1).to(self.device)

    def draw(self, ax=None):
        """
        Draw the task routing graph.
        """
        graph = self.G.copy()
        graph.add_edge("fix", "A0")
        graph.add_edge("stim", "A1")
        graph.add_edge("task", "A2")
        
        # Color rule for nodes
        def color_rule(node_name):
            if "A" in node_name: 
                if f"A{self.hparams.num_areas}" != node_name:
                    return "skyblue"
                else:
                    return "limegreen"
            else:
                return "salmon"
        color_map = [color_rule(node) for node in graph.nodes]
        
        # Draw graph
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=800, node_color=color_map, font_weight='bold', connectionstyle='arc3, rad = 0.1', ax=ax)
            
    @staticmethod
    def get_insert_func_nested(arr):
        """
        Get insert/slice functions for nested indices.
        
        Args:
            arr: nested list, where idx1 is the source area, and idx2 is the output area.

        Returns:
            insert_wrap: Function to insert data into the tensor.
            slice_wrap: Function to slice the tensor.
        """
        arr_flat = []
        for item in arr:
            arr_flat += item
        
        insert_func, _, slice_func = get_insert_func(arr_flat, return_slice=True)

        def get_idx(idx1, idx2):
            """
            Get the index of the tensor to insert data into.
            """
            sum_idx = 0
            for i in range(idx1):
                sum_idx += len(arr[i])
            sum_idx += len(arr[idx1][:idx2])
            return sum_idx

        def insert_wrap(tensor, data, idx1, idx2):
            """
            Insert data into the tensor.
            """
            converted_idx = get_idx(idx1, idx2)
            insert_func(tensor, data, converted_idx)
        
        def slice_wrap(tensor, idx1, idx2):
            """
            Slice the tensor.
            """
            converted_idx = get_idx(idx1, idx2)
            return slice_func(tensor, converted_idx)

        return insert_wrap, slice_wrap
    
    def _compute_ramp(self, start, increase):
        """
        Compute a coefficient that ramps from 0 to 1 over `increase` epochs.
        """
        ramp = (self.current_epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)