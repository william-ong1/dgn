import re
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso

flatten = lambda arr: arr.reshape(-1, arr.shape[-1])

def check_pattern(string, patterns):
    if len(patterns) == 0: return True

    regex_patterns = [re.escape(p) for p in patterns]
    full_regex = "|".join(regex_patterns)
    compiled_regex = re.compile(full_regex)
    return bool(compiled_regex.search(string))

def get_insert_func(sizes, return_slice=False):
    data_ends = np.cumsum(sizes)
    data_starts = np.insert(data_ends, 0, 0)[:-1]
    
    def insert_tensor(tensor, data, index):
        start, end = data_starts[index], data_ends[index]
        tensor[..., start:end] = data.clone()

    def exclude_tensor(tensor, index):
        start, end = data_starts[index], data_ends[index]
        indices_to_include = torch.tensor(list(range(start)) + list(range(end, data_ends[-1]))).to(torch.int64)
        sliced_tensor = torch.index_select(tensor, dim=-1, index=indices_to_include.to(tensor.device))
        return sliced_tensor
    
    def slice_tensor(tensor, index):
        start, end = data_starts[index], data_ends[index]
        indices_to_include = torch.tensor(list(range(start, end))).to(torch.int64)
        sliced_tensor = torch.index_select(tensor, dim=-1, index=indices_to_include.to(tensor.device))
        return sliced_tensor

    if not return_slice:
        return insert_tensor, exclude_tensor
    else:
        return insert_tensor, exclude_tensor, slice_tensor
    
class PolyRegression:
    def __init__(self, degree, alpha=0.0, tpe="ridge"):
        self.degree = degree
        self.alpha = alpha
        self.poly_features = PolynomialFeatures(degree=degree)
        
        if tpe == "ridge":
            self.reg = Ridge(alpha=alpha)
        elif tpe == "lasso":
            self.reg = Lasso(alpha=alpha)
        else:
            raise ValueError()

    def fit(self, X, y):
        X_poly = self.poly_features.fit_transform(X)
        self.reg.fit(X_poly, y)
        
    def ffit(self, X, y):
        self.fit(flatten(X), flatten(y))

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return self.reg.predict(X_poly)
    
    def fpredict(self, X):
        X_poly = self.poly_features.transform(flatten(X))
        return self.reg.predict(X_poly).reshape(*X.shape[:2], -1)

    def score(self, X, y): # X: prediction, y: true
        X_poly = self.poly_features.transform(X)
        return self.reg.score(X_poly, y)
    
    def fscore(self, X, y):
        pred = self.fpredict(X)
        return r2_score(flatten(y), flatten(pred))
    

Messages = namedtuple(
    "PlainMessages",
    [
        "mesgs",
        "latents",
    ],
)

PassDecisionMotionMessages = namedtuple(
    "PassDecisionMotionMessages",
    [
        "p_to_d",
        "d_to_m",
        "d",
        "a",
        "d_rep",
        "m_rep"
    ],
)
    
class RK4Solver:
    def __init__(self, dx_dt, t0, tf, dt):
        """
        Initializes the RK4 solver.
        
        Parameters:
            dx_dt (function): Function that computes the derivatives. Should take two arguments (x, t).
            t0 (float): Initial time.
            tf (float): Final time.
            dt (float): Time step for integration.
        """
        self.dx_dt = dx_dt
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        
    def solve(self, x0):
        """
        Solves the system of ODEs using the RK4 method.
        
        Parameters:
            x0 (np.ndarray): Initial condition for the system.
        
        Returns:
            t (np.ndarray): Array of time points.
            x (np.ndarray): Array of state variables at each time point.
        """
        num_steps = int((self.tf - self.t0) / self.dt)
        n = len(x0)
        
        t = np.linspace(self.t0, self.tf, num_steps + 1)
        x = np.zeros((num_steps + 1, n))
        x[0] = x0
        
        for i in range(num_steps):
            k1 = self.dt * self.dx_dt(x[i], t[i])
            k2 = self.dt * self.dx_dt(x[i] + 0.5 * k1, t[i] + 0.5 * self.dt)
            k3 = self.dt * self.dx_dt(x[i] + 0.5 * k2, t[i] + 0.5 * self.dt)
            k4 = self.dt * self.dx_dt(x[i] + k3, t[i] + self.dt)
            
            x[i + 1] = x[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return t, x
    
class Flatten:
    def __init__(self):
        self.shape = None
    
    def fit(self, arr):
        self.shape = arr.shape
        
    def __call__(self, arr):
        self.fit(arr)
        return arr.reshape(*self.shape[:-1], self.shape[-1])
    
    def restore(self, arr):
        return arr.reshape(*self.shape)

def expand_input_channel_size(
    channel,
    added_dims: int = 1,
    mean: float = 1.0,
    std: float = None,
):
    if added_dims <= 0: return channel  # nothing to do

    # Choose std so that P(X < 0) ≈ 0.1 for X ~ N(mean, std)
    # => mean / std ≈ 1.2816 (10th percentile z-score)
    if std is None:
        z_0_1 = 1.2815515655446004
        std = mean / z_0_1

    rnn_base = channel.rnn
    cell = rnn_base.model

    if not isinstance(cell, (nn.GRUCell, nn.RNNCell)):
        raise TypeError(
            f"expand_input_channel_size expects rnn.model to be GRUCell or RNNCell, "
            f"but got {type(cell)}"
        )

    with torch.no_grad():
        W = cell.weight_ih  # shape: (num_rows, old_in)
        num_rows, old_in = W.shape
        new_in = old_in + added_dims

        # New expanded weight_ih
        new_W = torch.empty(num_rows, new_in, device=W.device, dtype=W.dtype)
        # Copy existing columns
        new_W[:, :old_in] = W.data
        # Initialize new columns ~ N(mean, std)
        new_W[:, old_in:] = torch.normal(
            mean=mean,
            std=std,
            size=(num_rows, added_dims),
            device=W.device,
            dtype=W.dtype,
        )
        cell.weight_ih = nn.Parameter(new_W, requires_grad=True)

        # Update input_size attribute if present
        if hasattr(cell, "input_size"): cell.input_size = new_in

    # Update bookkeeping for wrappers
    channel.input_size += added_dims
    rnn_base.input_size += added_dims
    return channel


def generate_noisy_sine_waves(batch, time, dim, freq, noise_level=0.1):
    """
    Generate noisy sine waves with different frequencies for each dimension.

    Args:
        batch (int): Number of batches.
        time (int): Number of time steps.
        dim (int): Number of dimensions.
        freq (list): List of frequencies for each dimension.
        noise_level (float): Level of noise to be added to the sine waves.

    Returns:
        waves (numpy.ndarray): Array of shape (batch, time, dim) containing the generated noisy sine waves.
    """
    waves = np.zeros((batch, time, dim))
    time_vector = np.arange(time)

    for d in range(dim):
        for b in range(batch):
            noise = np.random.normal(scale=noise_level, size=time)
            phase = np.random.uniform(0, 2 * np.pi)
            wave = np.sin(freq[d] * time_vector + phase) + noise
            waves[b, :, d] = wave

    return waves