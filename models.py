import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

from utils.torch_utils import RNNChannel, MLPBase
from utils.common_utils import Messages, PassDecisionMotionMessages


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
