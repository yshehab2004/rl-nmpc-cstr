import numpy as np

from models.params import load_params

PARAMS = load_params()
_MEAS = PARAMS["measurement"]

C_B_INDEX = 1
T_R_INDEX = 2
MEASURED_INDICES = (T_R_INDEX, C_B_INDEX)


def noise_std(noise_frac=None):
    noise_frac = _MEAS["noise_frac"] if noise_frac is None else noise_frac
    return np.array([
        noise_frac * _MEAS["T_R_range"],
        noise_frac * _MEAS["C_b_range"],
    ])


def measurement_covariance(noise_frac=None):
    return np.diag(noise_std(noise_frac) ** 2)


def true_measurement(x):
    x = np.asarray(x, dtype=float).flatten()
    return np.array([x[T_R_INDEX], x[C_B_INDEX]])


def measure(x, rng=None, noise_frac=None):
    y = true_measurement(x)
    if rng is None:
        return y
    return y + rng.normal(0.0, noise_std(noise_frac))
