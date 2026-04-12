import torch
import torch.nn as nn
import pytorch_lightning as pl

from dgn.utils.torch_utils import RNNChannel, MLPBase
from dgn.utils.common_utils import Messages, PassDecisionMotionMessages

class DGNBase(pl.LightningModule):
    """
    Base class for all Data Generating Networks (DGN).
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
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

class MemoryNetwork(DGNBase):
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
                
            lag: Time-step lag between inter-regional communication.
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
            lr: Fixed learning rate.

            ext_input_dim: Dimensionality of external perturbation inputs.
                Perturbations are implemented as a step input from time steps
                100 to 120.
            ext_input_amp: Amplitude of perturbation inputs.
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
        
        # Setup external inputs (if applicable)
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
        assert self.hparams.channel_noise_type in ['fixed', 'variable']
        self.cnoise_weight_generator = VariableNoise(
            self.hparams.channel_noise,
            hps.total_mesgs,
            device = 'cuda',
            off = (self.hparams.channel_noise_type == 'fixed'),
        )
        self.h_noise_weight = torch.ones(hps.num_areas * hps.hidden_size).to(self.device) * hps.noise
        self.c_noise_weight = torch.ones(hps.total_mesgs).to(self.device) * hps.channel_noise
        
    def forward(self, inp, step_type):
        hps = self.hparams
        batch, time, _ = inp.shape # batch, time, total rank
        
        # Build storage
        self._build_save_var(batch, time) # build storage for mesgs, latents
        h = torch.zeros(hps.num_areas, batch, hps.hidden_size).to(self.device)
        mesgs = torch.zeros(batch, hps.total_mesgs).to(self.device) # batch, total mesg channels
        
        # Setup external input
        self.ext_inputs = {}
        for ia, area_name in enumerate(self.area_names):
            self.ext_inputs[area_name] = torch.zeros(batch, time, hps.ext_input_dim[ia], device=self.device)
            has_ext_inp = torch.bernoulli(torch.full((batch,), hps.ext_input_perc)) * hps.ext_input_amp # shape=(batch,)
            self.ext_inputs[area_name][:, 100:120] = torch.tile(has_ext_inp.reshape(-1, 1, 1), (1, 20, 1)).to(self.device)
        
        # Main loop
        for t in range(time):

            # Add perturbation to hidden states
            # This is correct because h is shaped in the order of
            # (hs of area 1, hs of area 2,...)
            h = h + torch.randn_like(h) * self.h_noise_weight.reshape(-1, 1, hps.hidden_size).to(self.device)
            
            # Setup variable storage per time t
            h_ias = []
            latents = torch.zeros(batch, hps.total_mesgs * hps.memory).to(self.device)
            mesgs_new = torch.zeros(batch, hps.total_mesgs).to(self.device)
            
            # Replace self-messages by input 
            # (it's okay because it's saved in previous time step)
            for ia in range(hps.num_areas):
                idx = ia * hps.num_areas + ia
                mesgs[..., self.get_index_mesg(idx)] = inp[:, t, self.get_index_rank(ia)].to(self.device)
            
            for ia, (area_name, area) in enumerate(self.areas.items()):
                
                # Indexing
                ibase = hps.num_areas * ia
                isource = torch.arange(hps.num_areas) + ibase
                
                # Forward pass through individual areas
                mesgs_inp = mesgs[..., self.get_index_mesg(*isource)]
                channel_noise = torch.randn_like(mesgs_inp) * self.c_noise_weight[self.get_index_mesg(*isource)].reshape(1, -1).to(self.device)
                mesgs_inp = mesgs_inp + channel_noise
                mesgs_inp = torch.cat([mesgs_inp, self.ext_inputs[area_name][:, t]], dim=-1).to(self.device) # add ext inp
                h_ia, mesgs_split = area(mesgs_inp, h[ia]) # area forward pass
                
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
                
        # Adjust noise weight
        self.h_noise_weight = self.noise_weight_generator(torch.cat([hs for hs in self.hidden_states.values()], dim=2))
        self.c_noise_weight = self.cnoise_weight_generator(self.save_var.mesgs)
                
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        inp, info = batch
        self.current_batch = inp
        self.current_info = info
            
        # Forward pass
        inp = inp[0]
        batch_size, time_size = inp.shape[:2]
        self.forward(inp, step_type)
        
        # Save input
        starts = [0] + list(hps.ranks)
        ends = list(torch.cumsum(hps.ranks, dim=0))
        # self.inputs = {}
        # for ia, area_name in enumerate(self.areas):
        #     self.inputs[area_name] = torch.cat([inp[..., starts[ia]:ends[ia]], self.ext_inputs[area_name]], dim=2)
        self.inputs = self.ext_inputs
        
        # Mesgs must reflect latent states of the source area
        inp_extended = []
        for ia in range(hps.num_areas):
            for ib in range(hps.num_areas):
                if hps.connectome[ia, ib]:
                    inp_extended.append(inp[..., self.get_index_rank(ib)])
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
        
        loss_hist = self.mseloss(self.save_var.latents[:, hps.memory + hps.lag*2:], inp_hist[:, :-hps.lag*2])
        loss_hist = torch.sum(loss_hist) / batch_size
        loss = loss_mesg + loss_hist
            
        # Log
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
            
class PassDecision(pl.LightningModule):
    def __init__(
        self,
        noise_p: float = 0.0,
        noise_d: float = 0.0,
        hidden_size: int = 64,
        lag: int = None,
        p_to_d_coef: float = 1.0,
        rep_coef: float = 0.0,
        binary_output: bool = True,
        rnn_nonlinearity: str = "tanh",
        lr: float = 4.0e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        hps = self.hparams
        
        # Fixed hps
        hps.input_dim = 2
        hps.channel_size = 2
        
        # Pass area: input + latent (dim=1)
        self.P_area = RNNChannel(hps.input_dim, hps.hidden_size, [hps.input_dim], 
                                 None, rnn_nonlinearity=hps.rnn_nonlinearity)
        # Decision area: input + latent (dim=1)
        self.D_area = RNNChannel(hps.input_dim, hps.hidden_size, [hps.channel_size], 
                                 None, rnn_nonlinearity=hps.rnn_nonlinearity)

        # Loss functions and decoders
        nonlinearity = "sigmoid" if hps.binary_output else None
        self.decoder = MLPBase([[hps.channel_size, hps.input_dim, nonlinearity]])
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCELoss() # expects probability (needs sigmoid)
        self.celoss = nn.CrossEntropyLoss() # does not expect sigmoid
        
        # Enforce representations of u in D
        self.D_decoder = MLPBase([[hps.hidden_size, hps.input_dim, None]])
        
    def forward(self, inp, latent, go, ctxt, step_type):
        hps = self.hparams
        batch, time, _ = inp.shape
        self._build_save_var(batch, time)

        h_p = torch.zeros(batch, hps.hidden_size).to(self.device)
        h_d = torch.zeros(batch, hps.hidden_size).to(self.device)
        
        for t in range(time):

            # Add perturbation to hidden states
            h_p = h_p + torch.randn_like(h_p) * hps.noise_p
            h_d = h_d + torch.randn_like(h_d) * hps.noise_d
            
            h_p, p_to_d = self.P_area(inp[:,t,:], h_p)
            h_d, d_to_m = self.D_area(p_to_d, h_d)
            
            d = self.decoder(d_to_m)
            d_rep = self.D_decoder(h_d)

            self.save_var.p_to_d[:,t] = p_to_d
            self.save_var.d_to_m[:,t] = d_to_m
            self.save_var.d[:, t] = d
            self.save_var.d_rep[:, t] = d_rep
            
            self.hidden_states["P"][:,t] = h_p
            self.hidden_states["D"][:,t] = h_d

        return list(self.save_var._asdict().values()) + list(self.hidden_states.values())
    
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        self.current_batch, info = batch
        self.current_info = info
        if not hps.lag:
            plag = nlag = None
        else:
            plag = hps.lag
            nlag = -hps.lag
        
        # Forward pass
        inp, latent, go, ctxt, action = self.current_batch
        p_to_d, d_to_m, d_pred, a, d_rep, m_rep, h_p, h_d = self.forward(inp, latent, go, ctxt, step_type)
        
        # Get necessary components
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

        # Log
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
        self.save_var = PassDecisionMotionMessages(
            p_to_d = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            d_to_m = torch.zeros(batch_size, time, hps.channel_size).to(self.device),
            d = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            a = torch.zeros(0).to(self.device),
            d_rep = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            m_rep = torch.zeros(0).to(self.device),
        )
        self.hidden_states = {
            "P": torch.zeros(batch_size, time, hps.output_size).to(self.device),
            "D": torch.zeros(batch_size, time, hps.output_size).to(self.device),
        }
        
    @staticmethod
    def init_weight(pm): nn.init.normal_(pm, mean=0.0, std=1.0)
            
class VariableNoise:
    def __init__(self, init_noise, feature_dim, ema_decay=0.99, device='cuda', off=False):
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
            return torch.full((F,),
                               self.init_noise,
                               device=data.device,
                               dtype=data.dtype)

        # Compute per-feature norms
        with torch.no_grad():
            # Norm across time per sample, then average across batch
            mean_sq = data.pow(2).mean(dim=(0, 1))  # shape: (F,)
            norm = torch.sqrt(mean_sq + 1e-8)       # per-feature magnitude

            # Update EMA
            if self.ema_norm.sum() == 0: # first update → direct assign
                self.ema_norm = norm
            else:
                self.ema_norm = (
                    self.ema_decay * self.ema_norm
                    + (1 - self.ema_decay) * norm
                )

        scaled_noise = self.ema_norm * self.noise_dist
        return scaled_noise
    
    
class ContextualRotation(DGNBase):
    def __init__(
        self,
        num_ctxt: int,
        f_x: dict,                    # area_name: {ctxt: A}
        f_u: dict,                    # tar: {src: {ctxt: B}}
        g_per_ctxt: bool = True,      # different projection per context
        lag: int = 2,
        noise_d: float = 0.0,
        noise_p: float = 0.0,
        hidden_size: int = 64,
        lr: float = 4.0e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['f_x', 'f_u'])
        hps = self.hparams
        self.area_names = list(f_x.keys())
        
        # Setup A
        hps.f_x = {}
        for an, Adict in f_x.items():
            hps.f_x[an] = {c: torch.Tensor(v).to(self.device) for c, v in Adict.items()}
        
        # Setup B
        hps.f_u = {}
        for tar in self.area_names:
            if tar in hps.f_u.keys():
                hps.f_u[tar] = {}
                for src, Bdict in hps.f_u[tar].items():
                    hps.f_u[tar][src] = {c: torch.Tensor(v).to(self.device) for c, v in Bdict.items()}
            else:
                hps.f_u[tar] = {}
                
        # Setup projection matrix g
        self.g = {}
        for an in self.area_names:
            self.g[an] = {
                c: 4*torch.rand(hps.hidden_size, 2, device=self.device)+1 for c in range(hps.num_ctxt)
            } # projection strength: [1,5)
            
        self.placeholder = nn.Linear(1, 1)
        
    def forward(
        self,
        batch,
    ):
        hps = self.hparams
        (ctxt,) = batch
        
        batch_dim, time_dim = ctxt.shape[:2]
        na = len(self.area_names)
        device = self.device
        
        # Storage
        x = torch.randn(na, batch_dim, 2, dtype=torch.float32, device=device)
        self.latents = {
            an: torch.empty(batch_dim, time_dim, 2, device=device)
            for an in self.area_names
        }
        self.hidden_states = {
            an: torch.empty(batch_dim, time_dim, hps.hidden_size, device=device)
            for an in self.area_names
        }
        
        for t in range(time_dim):
            c = ctxt[:, t, 0] # shape = (batch,)
            
            for ia, area_name in enumerate(self.area_names):
                
                # Evolve
                Adict = hps.f_x[area_name]
                A_c = self._get_matrix_by_context(c, Adict, batch_dim)
                x_next = torch.bmm(A_c, x[ia].unsqueeze(-1)).squeeze(-1)
                
                # Add input
                for src, Bdict in hps.f_u[area_name].items():
                    
                    isrc = int(src[1:])
                    if hps.lag == 0:
                        x_src = x[isrc]
                    elif t >= hps.lag:
                        x_src = self.latents[src][:, t-hps.lag]
                    else:
                        x_src = torch.zeros_like(x[isrc])
                    B_c = self._get_matrix_by_context(c, Bdict, batch_dim)
                    x_next += torch.bmm(B_c, x_src.unsqueeze(-1)).squeeze(-1)
                    
                # Add noise
                x_next += hps.noise_d * torch.randn_like(x_next)
                
                # Get projection
                gdict = self.g[area_name]
                g_c = self._get_matrix_by_context(c, gdict, batch_dim, output_dim=hps.hidden_size)
                h = torch.bmm(g_c, x_next.unsqueeze(-1)).squeeze(-1)
                h +=  hps.noise_p * torch.randn_like(h)
                
                # Store
                self.latents[area_name][:, t] = x_next
                self.hidden_states[area_name][:, t] = h
                
            x = torch.stack(
                [self.latents[an][:, t] for an in self.area_names],
                dim=0
            )
                
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        self.current_batch, self.current_info = batch
        
        # Forward pass
        self.forward(self.current_batch)
        return None
    
    def _get_matrix_by_context(self, ctxt, Mdict, batch_dim, output_dim=2):
        hps = self.hparams
        M_c = torch.empty(batch_dim, output_dim, 2, device=self.device)
        for nc in range(hps.num_ctxt):
            idx_c = (ctxt == nc)
            if sum(idx_c) > 0:
                M_c[idx_c] = Mdict[nc].to(self.device)
        return M_c