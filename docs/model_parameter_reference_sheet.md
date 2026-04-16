# Parameter Reference

This section lists the constructor parameters used by each model and data module.

---

### Memory Network model (`models.MemoryNetwork`)

Required:
- `ranks` (list): input dimensionality per area
- `connectome` (list[list]): binary connectivity matrix (target x source)

Optional:
- `lag` (int, default `1`): communication lag in timesteps
- `memory` (int, default `5`): how many past steps to encode
- `noise` (float, default `0.0`): hidden-state noise magnitude
- `channel_noise` (float, default `0.0`): communication channel noise magnitude
- `noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `channel_noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `hidden_size` (int, default `64`): recurrent hidden units per area
- `lr` (float, default `4e-3`): optimizer learning rate
- `ext_input_dim` (int or list, default `0`): external input dimensions per area
- `ext_input_amp` (int, default `-1`): amplitude of external perturbation
- `ext_input_perc` (float, default `0.0`): fraction of trials receiving perturbation

### Memory Network data module (`datamodules.NoisySources`)

Required:
- `batch_total` (int): total number of trials
- `time_total` (int): timesteps per trial
- `input_dim` (list or int): signal dimensions (list is summed internally)

Optional:
- `p_split` (list, default `[0.8, 0.2]`): train/val split ratio
- `batch_size` (int, default `64`): train batch size
- `mesg_type` (str or list, default `"white noise"`): `white noise`, `intg_noise`, `filter noise`, `sine wave`
- `mesg_kwargs` (dict, default `{}`): extra args (e.g., filter std)
- `cumsum` (bool, default `False`): legacy parameter (unused path)

---

### Pass Decision model (`models.PassDecision`)

Optional (all constructor args have defaults):
- `noise_p` (float, default `0.0`): noise in pass-area hidden states
- `noise_d` (float, default `0.0`): noise in decision-area hidden states
- `hidden_size` (int, default `64`): recurrent hidden units per area
- `lag` (int or `None`, default `None`): temporal shift for supervision alignment
- `p_to_d_coef` (float, default `1.0`): weight on pass-to-decision reconstruction loss
- `rep_coef` (float, default `0.0`): weight on representation loss (`d_rep_loss`)
- `binary_output` (bool, default `True`): binary (BCE) vs continuous (MSE) decision output
- `rnn_nonlinearity` (str, default `"tanh"`): recurrent nonlinearity
- `lr` (float, default `4e-3`): optimizer learning rate

Common config note:
- your YAML also sets `output_size`; this is used when allocating hidden-state export buffers.

### Pass Decision data module (`datamodules.LatentDecision`)

Required:
- `batch_total` (int): total number of trials
- `time_total` (int): timesteps per trial (`>= 200` required)
- `hidden_size` (int): model hidden size (kept aligned from config)

Optional:
- `decay_factor` (float, default `1.0`): decay used in latent filtering
- `latent_factor` (float, default `1.0`): latent amplitude scale
- `p_split` (list, default `[0.8, 0.2]`): train/val split ratio
- `batch_size` (int, default `64`): train batch size
- `lag` (int/bool, default `False`): lag used for trajectory generation alignment
- `mesg_dist` (str, default `"normal"`): `normal`, `exponential`, `uniform`, `sine wave`
- `binary_decision` (bool, default `False`): generate binary-style decision trajectories
- `sig_smooth` (float or `None`, default `None`): optional temporal smoothing sigma

---

### TTDGN / Multi-task reference (`ttdgn`)

#### Multi-task model (`models.MultiTaskNet`)

Required:
- `num_areas` (int): number of recurrent areas
- `task_names` (list): task IDs to train jointly

Optional:
- `hidden_size` (int, default `32`): hidden units per area
- `num_angles` (int, default `36`): discretization bins for response output
- `num_channels` (int, default `4`): communication channels per edge
- `noise` (float, default `0.0`): hidden-state noise magnitude
- `noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `channel_noise` (float, default `0.0`): communication noise magnitude
- `channel_noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `lr_init` (float, default `4e-3`): optimizer learning rate
- `angle_start_epoch` (int, default `50`): epoch where angle loss ramp starts
- `angle_increase_epoch` (int, default `100`): ramp duration for angle loss
- `angle_scale` (float, default `0.0`): max scaling for angle loss
- `l1_start_epoch` (int, default `150`): epoch where L1 ramp starts
- `l1_increase_epoch` (int, default `150`): ramp duration for L1 penalty
- `l1_scale` (float, default `0.0`): max scaling for communication L1 loss
- `delay` (int, default `0`): communication delay in timesteps
- `diagram` (list or `None`, default `None`): explicit graph edges; if provided, bypass random graph search
- `sacc_scale` (float, default `1.0`): scaling for fixation/saccade loss
- `sacc_output_areas` (list or `None`, default `None`): which areas contribute to saccade output
- `stim_input_areas` (list or `None`, default `None`): area routing for `[fix, stim1, stim2, task]` channels
- `graph_kwargs` (dict, default `{}`): graph generation settings

Critical note:
- if `diagram` is `None`, `graph_kwargs` must include:
  - `perc_conns` (fraction of possible edges)
  - `shortest` (max path length constraint from stimulus-input areas to output sub-area)

#### Multi-task data module (`datamodules.MultiTask`)

Required:
- `task_names` (list): task names from `task_map`
- `batch_total` (int): total number of trials
- `time_total` (int): timesteps per trial

Optional:
- `p_split` (list, default `[0.8, 0.2]`): train/val split ratio
- `batch_size` (int, default `64`): train batch size
- `train_type` (str, default `"random"`): task scheduling mode (`random`, `batch_uniform`, `curriculum_ratio`, ...)
- `train_type_kwargs` (dict, default `{}`): schedule-specific args
- `dm_seed` (int, default `0`): datamodule RNG seed
- `noise_sig` (float, default `0.0`): noise level added to generated task signals