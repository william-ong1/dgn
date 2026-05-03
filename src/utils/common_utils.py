import numpy as np
import torch
from collections import namedtuple 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso


# Helper functions
sigmoid = lambda x: 1 / (1 + np.exp(-x)) # sigmoid function
normalize = lambda theta: np.arctan2(np.sin(theta), np.cos(theta)) # convert angle into interval [-pi, pi]
normalize_torch = lambda theta: torch.arctan2(torch.sin(theta), torch.cos(theta))
flatten = lambda arr: arr.reshape(-1, arr.shape[-1]) # flatten the tensor


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

    # Generate the sine waves
    for d in range(dim):
        for b in range(batch):
            noise = np.random.normal(scale=noise_level, size=time)
            phase = np.random.uniform(0, 2 * np.pi)
            wave = np.sin(freq[d] * time_vector + phase) + noise
            waves[b, :, d] = wave

    return waves


def area_activity_to_poisson_counts(
    activity: np.ndarray,
    dt: float,
    rate_max: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Map continuous hidden activity to Poisson spike counts per bin.

    Args:
        activity: Hidden state activity.
        dt: Time bin in seconds.
        rate_max: Maximum rate (Hz).
        rng: NumPy random generator for draws. If ``None``, one is built from ``seed``.
        seed: Used only when ``rng`` is ``None`` (via ``np.random.default_rng(seed)``).

    Returns:
        Spike counts as ``float32`` (same shape as ``activity``).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    rates = rate_max * (activity + 1.0) / 2.0
    lam = rates * dt
    lam = np.clip(lam, a_min=0.0, a_max=None)
    counts = rng.poisson(lam)
    return counts.astype(np.float32)


# Named tuples for message storage for various DGN models
Messages = namedtuple(
    "PlainMessages",
    [
        "mesgs",
        "latents",
    ],
)


# Named tuples for message storage for PassDecision model
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


def get_insert_func(arr_flat, return_slice=False):
    """
    Build insert/slice helpers for a 2D tensor ``(batch, sum(arr_flat))``.

    Args:
        arr_flat: List of segment widths along the last dimension.
        return_slice: If True, return ``(insert_func, exclude_func, slice_func)``.

    Returns:
        Tuple of callables. ``insert_func(tensor, data, idx)`` writes ``data`` into
        the segment at flat index ``idx``. ``slice_func(tensor, idx)`` returns that segment.
    """
    if not return_slice:
        raise NotImplementedError("Only return_slice=True is supported.")
    arr_flat = [int(w) for w in arr_flat]
    offsets = [0]
    for w in arr_flat:
        offsets.append(offsets[-1] + w)

    def insert_func(tensor, data, idx):
        start = offsets[idx]
        tensor[:, start : start + data.shape[-1]] = data

    def exclude_func(*args, **kwargs):
        raise NotImplementedError("exclude_func is not used for this DGN layout.")

    def slice_func(tensor, idx):
        start = offsets[idx]
        end = start + arr_flat[idx]
        return tensor[:, start:end]

    return insert_func, exclude_func, slice_func


def categorize(data, itvl, num_angles):
    """
    Categorize the data into the number of angles residing within the interval. 
    """
    bins = torch.linspace(*itvl, num_angles + 1).to(data.device)
    return (torch.bucketize(data, bins) - 1).reshape(*data.shape)


def one_hot_encode(data, itvl, num_angles):
    """
    One hot encode the data into the number of angles residing within the interval.
    """
    if num_angles is None: # should infer num_angles from data
        num_angles = int(torch.max(data)) + 1
    data = categorize(data, itvl, num_angles)

    # One hot encode the data
    one_hot = torch.zeros(data.shape[:-1] + (num_angles,)).to(data.device)
    one_hot[torch.arange(data.shape[0])[:, None], torch.arange(data.shape[1]), data.squeeze()] = 1
    return one_hot


def sum_nested(arr):
    """
    Sum uneven nested arrays.
    """
    total = 0
    for item in arr: total += sum(item)
    return total


# Helper class to perform polynomial regression for decodability analyses
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