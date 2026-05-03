import torch
import numpy as np
from collections import namedtuple 


# Helper functions
sigmoid = lambda x: 1 / (1 + np.exp(-x))
normalize = lambda theta: np.arctan2(np.sin(theta), np.cos(theta))


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
