import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, random_split
from scipy.ndimage import gaussian_filter1d
from utils.common_utils import generate_noisy_sine_waves, normalize
from utils.cognitive_tasks import task_map


class DGNDataModuleBase(pl.LightningDataModule):
    """
    Base class for all Data Generating Networks (DGN) Lightning DataModules.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def setup(self, stage=None):
        raise NotImplementedError("setup must be implemented in subclass!")

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            num_workers=16,
        )


class BasicDataset(Dataset):
    """
    Basic dataset class for Data Generating Networks (DGN).
    """
    def __init__(self, *iter_data, **kwargs):
        self.iter_data = iter_data
        self.info = kwargs

    def __getitem__(self, index):
        return tuple([item[index] for item in self.iter_data]), self.info

    def __len__(self):
        return len(self.iter_data[0])


class NoisySources(DGNDataModuleBase):
    """
    DataModule for generating noisy sources, used with the `MemoryNetwork` model.
    """
    def __init__(
        self,
        batch_total: int,
        time_total: int,
        input_dim: list | int,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        mesg_type: str = "white noise",
        mesg_kwargs: dict = {},
        sig_smooth: float = None,
        resultpath: str = ".",
    ):
        """
        Args:
            batch_total: Total number of batches (trials).
            time_total: Total number of time steps.
            input_dim: Dimensionality of input for each area. If an integer 
                is passed, all areas will share the same dimensionality.
            p_split: Train/validation split percentage, must sum up to 1.
            batch_size: Size of each training batch. Validation batch size is always
                the total number of validation batches.
            mesg_type: Distribution type for signals. Options are:
                'white noise', 'intg noise', 'filter noise' or 'sine wave'.
                See `transform()` function for each option.
            mesg_kwargs: Corresponding kwargs for the chosen `mesg_type`.
            sig_smooth: Standard deviation for Gaussian smoothing.
            resultpath: Path to save the results.
        """
        super().__init__()
        self.save_hyperparameters()
        hps = self.hparams
        
        if isinstance(hps.input_dim, list):
            hps.input_dim = sum(hps.input_dim)
        
    def setup(self, stage=None):
        """
        Setup the dataset and split into train and validation sets.
        """
        hps = self.hparams

        # Generate input for sine wave message type
        if hps.mesg_type == 'sine wave':
            freq = np.random.uniform(0.1, 1.0, size=hps.input_dim)
            inp = generate_noisy_sine_waves(hps.batch_total, hps.time_total, hps.input_dim, freq, noise_level=0.0)
        else:
            # Generate white noise for all other message types
            inp = np.random.normal(size=(hps.batch_total, hps.time_total, hps.input_dim))
            
            # Preprocess message type
            if isinstance(hps.mesg_type, str):
                hps.mesg_type = [hps.mesg_type] * hps.input_dim
            
            # Change noise index by index according to mesg_type
            for idim, mtype in enumerate(hps.mesg_type): 
                inp = self.transform(inp, idim, mtype)
        
        # Apply smoothing if specified
        if hps.sig_smooth:
            inp = gaussian_filter1d(inp, sigma=hps.sig_smooth, axis=1)

        # Construct the dataset and split into train and validation sets
        ds = BasicDataset(
            inp.astype(np.float32),
        )
        self.train_ds, self.val_ds = random_split(ds, hps.p_split)
        
    def transform(self, x, idx, mesg_type):
        """
        Transform the input according to the message type.

        Args:
            x (numpy.ndarray): Input array.
            idx (int): Index of the input dimension to transform.
            mesg_type (str): Message type.

        Returns:
            x (numpy.ndarray): Transformed input array.
        """
        hps = self.hparams
        if mesg_type == "white noise": # no changes
            pass 
        elif mesg_type == "intg_noise": # performs cumsum
            x[..., idx] = np.cumsum(x[..., idx], axis=1)
        elif mesg_type == "filter noise": # gaussian filter
            assert 'std' in hps.mesg_kwargs.keys()
            x[..., idx] = gaussian_filter1d(x[..., idx], hps.mesg_kwargs['std'][idx], axis=1)
        else:
            raise ValueError()
        return x


class LatentDecision(DGNDataModuleBase):
    """
    DataModule for generating latent decisions, used with the `PassDecision` model.
    """
    def __init__(
        self,
        batch_total: int,
        time_total: int,
        decay_factor: float = 1.0,
        latent_factor: float = 1.0,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        lag: int = 0,
        mesg_dist: str = "normal",
        binary_decision: bool = False,
        sig_smooth: float = None,
        resultpath: str = ".",
    ):
        """
        Args:
            batch_total: Total number of batches (trials).
            time_total: Total number of time steps.
            decay_factor: Decay factor for latent signal filtering.
            latent_factor: Factor for latent signal amplitude scaling.
            p_split: Train/validation split percentage, must sum up to 1.
            batch_size: Size of each training batch. Validation batch size is always
                the total number of validation batches.
            lag: Time-step lag for trajectory generation alignment.
            mesg_dist: Distribution type for input signal. Options are:
                'normal', 'exponential', 'uniform', 'sine wave'.
            binary_decision: Whether to generate binary-style decision trajectories.
            sig_smooth: Standard deviation for Gaussian smoothing.
            resultpath: Path to save the results.
        """
        super().__init__()
        self.save_hyperparameters()
        assert time_total >= 200 # latest cue time 150, smallest action time 50
        
        # Fixed hparams
        hps = self.hparams
        hps.input_dim = 2
        
    def setup(self, stage=None):
        """
        Setup the dataset and split into train and validation sets.
        """
        hps = self.hparams

        # Generate input signals
        if hps.mesg_dist == "normal":
            inp = np.random.normal(size=(hps.batch_total, hps.time_total, hps.input_dim)) * 3
        elif hps.mesg_dist == "exponential":
            inp = np.random.exponential(scale=3.0, size=(hps.batch_total, hps.time_total, hps.input_dim)) - 3.0
        elif hps.mesg_dist == "uniform":
            inp = np.random.uniform(-5., 5., size=(hps.batch_total, hps.time_total, hps.input_dim))
        elif hps.mesg_dist == "sine wave":
            freq = np.random.uniform(0.1, 1.0, size=hps.input_dim)
            inp = generate_noisy_sine_waves(hps.batch_total, hps.time_total, hps.input_dim, freq, noise_level=0.5) * 4.
        else:
            raise ValueError(f"Invalid message distribution: {hps.mesg_dist}")
        
        # Apply smoothing if specified
        if hps.sig_smooth:
            inp = gaussian_filter1d(inp, sigma=hps.sig_smooth, axis=1)
            
        # Generate latent signal 
        latent = np.random.normal(size=(hps.batch_total, hps.time_total, 1))
        latent = self.exponential_filter(latent, hps.decay_factor, 1) * hps.latent_factor

        # Generate decision signal
        dec = np.cumsum(inp, axis=1)
        
        # Random sample go time, go cue duration = 10
        go_times = np.random.randint(low=100, high=150, size=hps.batch_total)
        go = np.zeros((hps.batch_total, hps.time_total, 1))
        for b, go_time in enumerate(go_times):
            go[b, go_time: go_time + 10, 0] = np.ones(10)
            
        # Generate context signal (at the start of the trial, 10-20 timestep)
        ctxt = np.zeros((hps.batch_total, hps.time_total, 1))
        for b, go_time in enumerate(go_times):
            ctxt[b, 10:20, 0] = np.ones(10) * np.random.choice([1, -1])
        
        # Generate trajectories
        action = np.zeros((hps.batch_total, hps.time_total, 3))
        ang_velocity = np.pi / (hps.time_total - 100) # doesn't turn more than half a circle (pi)

        lag = hps.lag
        for b in range(hps.batch_total):
            
            # Generate trajectory for continuous decision
            if not hps.binary_decision:
                traj = self.gen_trajectory_3d(
                    *dec[b, go_times[b] - lag],
                    latent[b, go_times[b] - lag].item(),
                    hps.time_total - go_times[b],
                    ang_velocity,
                    np.sign(ctxt[b].mean()),
                )
            else: # generate trajectory for binary decision
                traj = self.gen_trajectory_3d(
                    * np.sign(dec[b, go_times[b] - lag]) * 3, # amplitude=3
                    0,                                        # does not use latent
                    hps.time_total - go_times[b],
                    ang_velocity,
                    np.sign(ctxt[b].mean()),
                )
                
            # Pad trajectory with zeros to match total time
            traj_embed = np.pad(traj, ((go_times[b], 0), (0,0)), mode="constant", constant_values=0)
            action[b] = traj_embed
        
        # Construct the dataset and split into train and validation sets
        ds = BasicDataset(
            inp.astype(np.float32), 
            latent.astype(np.float32), 
            go.astype(np.float32), 
            ctxt.astype(np.float32), 
            action.astype(np.float32), 
        )
        self.train_ds, self.val_ds = random_split(ds, hps.p_split)
        
    @staticmethod
    def gen_trajectory_3d(x0, y0, z0, time, v, ctxt):
        """
        Generate the 3D trajectory points for a decision signal.

        Args:
            x0 (float): x dimension of the decision signal.
            y0 (float): y dimension of the decision signal.
            z0 (float): z dimension of the decision signal.
            time (int): time steps.
            v (float): angular velocity.
            ctxt (int): context signal.

        Returns:
            traj (numpy.ndarray): 3D trajectory points (time, 3)
        """
        sigmoid = lambda x: 1/(1+np.exp(-0.02 * x)) # scaled sigmoid function
        
        # Calculate radius and signs
        if ctxt == 1: # use x dimension
            r = sigmoid(np.sqrt(x0**2 + z0**2))
            sign = np.sign(x0)
        else: # use y dimension
            r = sigmoid(np.sqrt(y0**2 + z0**2))
            sign = np.sign(y0)

        # Calculate angles
        angles = np.arange(0, time) * v
        phi0 = (1-sign) * np.pi / 2

        # Calculate trajectory points
        if ctxt == 1:
            x_traj = r * np.cos(phi0 + angles)
            y_traj = np.zeros_like(x_traj)
        else:
            y_traj = r * np.cos(phi0 + angles)
            x_traj = np.zeros_like(y_traj)
        
        # Calculate z dimension
        z_traj = r * np.sin(phi0 + angles)
        traj = np.vstack([x_traj, y_traj, z_traj])
        
        return traj.T
    
    @staticmethod
    def exponential_filter(array, alpha, axis):
        """
        Apply a decaying exponential filter to an array along a specified dimension.

        Args:
            array (numpy.ndarray): Input array.
            alpha (float): Decay factor, where 0 < alpha < 1.
            axis (int): Dimension along which to apply the filter.

        Returns:
            numpy.ndarray: Filtered array.
        """
        def filter_func(data):
            decay_weights = alpha ** np.arange(len(data))[::-1]
            return np.convolve(data, decay_weights, mode='full')[:len(data)]

        return np.apply_along_axis(filter_func, axis, array)


class MultiTask(DGNDataModuleBase):
    """
    DataModule for generating multi-task data, used with the `MultiTaskNet` model.
    """
    def __init__(
        self,
        task_names: list,
        batch_total: int,
        time_total: int,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        train_type: str = "random",
        train_type_kwargs: dict = {},
        dm_seed: int = 0,
        noise_sig: float = 0.0,
        resultpath: str = ".",
    ):
        """
        Args:
            task_names: List of task names.
            batch_total: Total number of batches (trials).
            time_total: Total number of time steps.
            p_split: Train/validation split percentage, must sum up to 1.
            batch_size: Size of each training batch. Validation batch size is always
                the total number of validation batches.
            train_type: Type of training. Options are:
                'random', 'batch_uniform', 'curriculum_ratio'.
            train_type_kwargs: Keyword arguments for the training type.
            dm_seed: Seed for the data module.
            noise_sig: Standard deviation for noise.
            resultpath: Path to save the results.
        """
        super().__init__()
        self.save_hyperparameters()
        self.current_epoch = 0
        
    def setup(self, stage=None):
        """
        Setup the dataset and split into train and validation sets.
        """
        hps = self.hparams
        
        # Construct all "Task" classes
        tasks = []
        for task_name in hps.task_names:
            task_class, task_config = task_map[task_name]
            task = task_class(task_config, hps.time_total)
            tasks.append(task)
            try:
                task.draw(save=True, figname=task_name, n_batches=4)
            except:
                print('Task draw() failed. Continuing...')
        self.tasks = tasks
            
        # Generate train and validation sizes
        batch_train = int(hps.batch_total * hps.p_split[0])
        batch_val = hps.batch_total - batch_train
        
        # Generate validation set (random order)
        self.val_batch_order = self.random_order(batch_val)
        self.val_ds = self.gen_data(tasks, self.val_batch_order)
        
        # Generate train set
        self.gen_train_ds(batch_train)
        
    def gen_train_ds(self, batch_train):
        """
        Generate the train dataset according to the train type.
        """
        hps = self.hparams
        if hps.train_type == "random": # random order
            train_batch_order = self.random_order(batch_train)
        elif hps.train_type == "batch_uniform": # single task per batch
            train_batch_order = self.single_task_per_batch_order(batch_train, **hps.train_type_kwargs,)
        elif hps.train_type == "curriculum_ratio": # task ratio changes over time
            train_batch_order = self.curriculum_ratio_order(batch_train, **hps.train_type_kwargs,)
        else:
            raise ValueError(f"Unknown train_type: {hps.train_type}")
            
        # Generate train dataset
        self.train_batch_order = train_batch_order
        self.train_ds = self.gen_data(self.tasks, self.train_batch_order)
        
    def gen_data(self, tasks, batch_order):
        """
        Generate data for a given batch order.
        """
        hps = self.hparams

        def gen_noise(arr, mag=1):
            return np.random.normal(0, 1, arr.shape) * mag * hps.noise_sig

        batch_size = len(batch_order)

        # Initialize arrays for each task type
        fixs = np.zeros((batch_size, hps.time_total, 1))
        stim1s = np.zeros((batch_size, hps.time_total, 1))
        stim2s = np.zeros((batch_size, hps.time_total, 1))
        resps = np.zeros((batch_size, hps.time_total, 1))
        saccs = np.zeros((batch_size, hps.time_total, 1))
        amp1s = np.zeros((batch_size, 1))
        amp2s = np.zeros((batch_size, 1))
        
        # Generate data for each batch
        for b, tpe in enumerate(batch_order):
            fix, (stim1, amp1), (stim2, amp2), resp, sacc = tasks[int(tpe)].gen_single_trial()
            fixs[b, :] = fix + gen_noise(fix)
            stim1s[b, :] = normalize(stim1 + gen_noise(stim1, mag=0.1)) # stim1
            stim2s[b, :] = normalize(stim2 + gen_noise(stim2, mag=0.1)) # stim2
            resps[b, :] = resp
            saccs[b, :] = sacc
            amp1s[b] = amp1
            amp2s[b] = amp2
        
        # Create task indices and return dataset
        task_idxs = np.tile(batch_order.reshape(-1, 1, 1), (1, hps.time_total, 1))
        return BasicDataset(fixs, stim1s, amp1s, stim2s, amp2s, task_idxs, resps, saccs)
            
    def random_order(self, batch_size, **kwargs):
        """
        Repeated shuffled permutations of all task ids.
        """
        hps = self.hparams
        num_task = len(hps.task_names)
        task_idxs = np.arange(num_task).astype(int)
        
        rng = np.random.default_rng(hps.dm_seed)
        num_mini_batch = int(np.ceil(batch_size/num_task))
        batch = np.zeros(num_mini_batch * num_task)
        
        # Shuffle task indices for each mini-batch
        for b in range(num_mini_batch):
            rng.shuffle(task_idxs)
            batch[b * num_task: (b+1) * num_task] = task_idxs
        
        return batch[:batch_size]
    
    def single_task_per_batch_order(self, batch_total, **kwargs):
        """
        Long runs of one task (``batch_size * persist`` samples), then the next task, repeating.
        """
        defaults = {'persist': 1}
        defaults.update(kwargs)
        persist = defaults['persist']
        
        hps = self.hparams
        num_task = len(hps.task_names)
        task_idxs = np.arange(num_task).astype(int)
        
        # Tile task indices to create a base pattern
        base = np.tile(task_idxs.reshape(1, -1), (hps.batch_size * persist, 1)) # shape = (bs & persist, num_tasks)
        base = base.flatten(order='F') # [0, 0, 0... 1, 1, 1....,]
        num_repeats = int(np.ceil(batch_total / len(base)))
        return np.tile(base, num_repeats)[:batch_total]
    
    def curriculum_ratio_order(self, batch_total: int, **kwargs):
        """
        Per-block task draws with weights moving from uniform toward ``final_ratio`` as ``current_epoch`` increases.
        """
        defaults = {
            "persist": 1,
            "final_ratio": None,   # if None -> stays uniform
            "decay": 50.0,
        }
        defaults.update(kwargs)
        persist = int(defaults["persist"])
        final_ratio = defaults["final_ratio"]
        decay = float(defaults["decay"])

        hps = self.hparams
        num_task = len(hps.task_names)
        epoch = getattr(self, "current_epoch", 0)

        # Define initial (uniform) and target ratios
        init_ratio = np.ones(num_task, dtype=float) / num_task

        # Define target ratio and ensure it matches the number of tasks
        if final_ratio is None:
            target_ratio = init_ratio.copy()
        else:
            target_ratio = np.asarray(final_ratio, dtype=float)
            assert target_ratio.shape[0] == num_task, \
                f"final_ratio length {target_ratio.shape[0]} must match num_tasks={num_task}"
            target_ratio = target_ratio / target_ratio.sum()

        # alpha ~ 0 => uniform, alpha ~ 1 => target_ratio
        if decay <= 0:
            alpha = 1.0
        else:
            alpha = 1.0 - np.exp(-epoch / decay)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Interpolate between initial and target ratios
        curr_ratio = (1.0 - alpha) * init_ratio + alpha * target_ratio
        curr_ratio = curr_ratio / curr_ratio.sum()

        # Sample block-wise tasks according to curr_ratio
        block_size = hps.batch_size * persist
        num_blocks = int(np.ceil(batch_total / block_size))

        # Make RNG depend on epoch so pattern changes across epochs
        rng = np.random.default_rng(hps.dm_seed + int(epoch))

        # Choose a task id for each block
        block_tasks = rng.choice(num_task, size=num_blocks, p=curr_ratio)
        batch = np.repeat(block_tasks, block_size)
        return batch[:batch_total]

    def train_dataloader(self, shuffle: bool = False):
        # Batch order is produced by train_type (curriculum, etc.)
        return super().train_dataloader(shuffle=shuffle)
