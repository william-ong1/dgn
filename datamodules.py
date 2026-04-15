import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset, random_split
from scipy.ndimage import gaussian_filter1d

import paths as path
from utils.common_utils import generate_noisy_sine_waves

class DGNDataModuleBase(pl.LightningDataModule):
    """
    Base class for all Data Generating Networks (DGN) DataModules.
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


class NoisySources(DGNDataModuleBase):
    def __init__(
        self,
        batch_total: int,
        time_total: int,
        input_dim: list,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        mesg_type: str = "white noise",
        mesg_kwargs: dict = {},
        
        cumsum: bool = False, # legacy feature, to be removed
    ):
        """
        Args:
            batch_total: Total number of batches (trials)
            time_total: Total number of time steps
            input_dim: Dimensionality of input for each area. If an integer 
                is passed, all areas will share the same dimensionality.
            p_split: train/validation split percentage, must sum up to 1.
            batch_size: Size of each training batch. Validation batch size is always
                the total number of validation batches.
            mesg_type: Distribution type for signals. Options are:
                'white noise', 'intg noise', 'filter noise' or 'sine wave'.
                See `transform()` function for each option.
            mesg_kwargs: Corresponding kwargs for the chosen `mesg_type`.
        """
        super().__init__()
        self.save_hyperparameters()
        hps = self.hparams
        
        if isinstance(hps.input_dim, list): hps.input_dim = sum(input_dim)
        
    def setup(self, stage=None):
        hps = self.hparams

        if hps.mesg_type == 'sine wave':
            freq = np.random.uniform(0.1, 1.0, size=hps.input_dim)
            inp = generate_noisy_sine_waves(hps.batch_total, hps.time_total, hps.input_dim, freq, noise_level=0.0)
        else:
            
            # Generate white noise
            inp = np.random.normal(size=(hps.batch_total, hps.time_total, hps.input_dim))
            
            # Preprocess
            if isinstance(hps.mesg_type, str):
                hps.mesg_type = [hps.mesg_type] * hps.input_dim
                
            # Change noise index by index according to mesg_type
            for idim, mtype in enumerate(hps.mesg_type): 
                inp = self.transform(inp, idim, mtype)
        
        ds = BasicDataset(
            inp.astype(np.float32),
        )
        self.train_ds, self.val_ds = random_split(ds, hps.p_split)
        
    def transform(self, x, idx, mesg_type):
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
    def __init__(
        self,
        batch_total: int,
        time_total: int,
        hidden_size: int,
        decay_factor: float = 1.,
        latent_factor: float = 1.,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        lag: int = False,
        mesg_dist: str = "normal",
        binary_decision: bool = False,
        sig_smooth: float = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert time_total >= 200 # latest cue time 150, smallest action time 50
        
        # Fixed hparams
        hps = self.hparams
        hps.input_dim = 2
        
    def setup(self, stage=None):
        hps = self.hparams

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
            raise ValueError()
            
        if hps.sig_smooth:
            inp = gaussian_filter1d(inp, sigma=hps.sig_smooth, axis=1)
            self.draw(inp)
            
        latent = np.random.normal(size=(hps.batch_total, hps.time_total, 1))
        latent = self.exponential_filter(latent, hps.decay_factor, 1) * hps.latent_factor
        dec = np.cumsum(inp, axis=1)
        
        # Random sample go time, go cue duration = 10
        go_times = np.random.randint(low=100, high=150, size=hps.batch_total)
        go = np.zeros((hps.batch_total, hps.time_total, 1))
        for b, go_time in enumerate(go_times):
            go[b, go_time: go_time + 10, 0] = np.ones(10)
            
        # Get context (at the start of the trial, 10-20 timestep)
        ctxt = np.zeros((hps.batch_total, hps.time_total, 1))
        for b, go_time in enumerate(go_times):
            ctxt[b, 10:20, 0] = np.ones(10) * np.random.choice([1, -1])
            
        # Generate rotation matrix, where R.T is inverse rotation matrix
        # R = self.gen_random_rotation_matrix(hps.hidden_size)
        
        # Generate trajectories
        action = np.zeros((hps.batch_total, hps.time_total, 3))
        ang_velocity = np.pi / (hps.time_total - 100) # doesn't turn more than half a circle (pi)
        lag = 0 if not hps.lag else hps.lag
        for b in range(hps.batch_total):
            
            if not hps.binary_decision:
                traj = self.gen_trajectory_3d(
                    *dec[b, go_times[b] - lag],
                    latent[b, go_times[b] - lag].item(),
                    hps.time_total - go_times[b],
                    ang_velocity,
                    np.sign(ctxt[b].mean()),
                )
            else:
                traj = self.gen_trajectory_3d(
                    * np.sign(dec[b, go_times[b] - lag]) * 3, # amplitude=3
                    0,                                        # does not use latent
                    hps.time_total - go_times[b],
                    ang_velocity,
                    np.sign(ctxt[b].mean()),
                )
                
            traj_embed = np.pad(traj, ((go_times[b], 0), (0,0)), mode="constant", constant_values=0) # shape = (total time, hidden size)
            action[b] = traj_embed
        
        ds = BasicDataset(
            inp.astype(np.float32), 
            latent.astype(np.float32), 
            go.astype(np.float32), 
            ctxt.astype(np.float32), 
            action.astype(np.float32), 
            # R=R.astype(np.float32),
        )
        self.train_ds, self.val_ds = random_split(ds, hps.p_split)
        
    @staticmethod
    def gen_trajectory_3d(x0, y0, z0, time, v, ctxt):
        sigmoid = lambda x: 1/(1+np.exp(-0.02 * x)) # Uses a scaled sigmoid function
        
        if ctxt == 1:
            r = sigmoid(np.sqrt(x0**2 + z0**2))
            sign = np.sign(x0)
        else:
            r = sigmoid(np.sqrt(y0**2 + z0**2))
            sign = np.sign(y0)

        angles = np.arange(0, time) * v
        phi0 = (1-sign) * np.pi / 2

        # Calculate trajectory points
        if ctxt == 1:
            x_traj = r * np.cos(phi0 + angles)
            y_traj = np.zeros_like(x_traj)
        else:
            y_traj = r * np.cos(phi0 + angles)
            x_traj = np.zeros_like(y_traj)
            
        z_traj = r * np.sin(phi0 + angles)
        traj = np.vstack([x_traj, y_traj, z_traj])
        
        return traj.T
    
    @staticmethod
    def gen_random_rotation_matrix(n):
        # Generate a random vector in n-dimensional space
        v = np.random.rand(n)

        # Construct skew-symmetric matrix
        V = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                V[i, j] = -v[j]
                V[j, i] = v[j]

        # Generate rotation matrix using Cayley transform
        R = np.eye(n) + 2 * np.dot(np.linalg.inv(np.eye(n) - V), V)

        return R
    
    @staticmethod
    def exponential_filter(array, alpha, axis):
        """Apply a decaying exponential filter to an array along a specified dimension.

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
    
    def draw(self, arr):
        batch, time, fea = arr.shape
        fig, axs = plt.subplots(fea, 4, figsize=(10, 3*fea))
        for i in range(fea):
            for b in range(4):
                axs[i, b].plot(arr[b, :, i], "b")
        plt.tight_layout()
        plt.savefig(os.path.join(path.resultpath, "results", "input.png"))

class BasicDataset(Dataset):
    def __init__(self, *iter_data, **kwargs):
        self.iter_data = iter_data
        self.info = kwargs

    def __getitem__(self, index):
        return tuple([item[index] for item in self.iter_data]), self.info

    def __len__(self):
        return len(self.iter_data[0])