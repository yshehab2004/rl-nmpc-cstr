import numpy as np

from models.params import load_params

PARAMS = load_params()


class StepDisturbance:

    def __init__(self, base: float, magnitude: float, t_start: float):
        self.base = base
        self.magnitude = magnitude
        self.t_start = t_start

    def __call__(self, t_now: float) -> float:
        return self.base + self.magnitude if t_now >= self.t_start else self.base


class RampDisturbance:

    def __init__(self, base: float, magnitude: float, t_start: float, t_end: float):
        self.base = base
        self.magnitude = magnitude
        self.t_start = t_start
        self.t_end = t_end

    def __call__(self, t_now: float) -> float:
        if t_now <= self.t_start:
            return self.base
        if t_now >= self.t_end:
            return self.base + self.magnitude
        frac = (t_now - self.t_start) / (self.t_end - self.t_start)
        return self.base + self.magnitude * frac


class OUNoiseDisturbance:

    def __init__(self, base, std: float, tau_h: float, seed: int,
                 t_max: float, grid_per_tau: int = 20):
        if tau_h <= 0:
            raise ValueError("tau_h must be positive: it is a correlation time.")
        self.base = base
        self.std = float(std)
        self.tau_h = float(tau_h)
        self.seed = int(seed)
        self.t_max = float(t_max)

        dt = self.tau_h / grid_per_tau
        n = int(np.ceil(self.t_max / dt)) + 2
        self._t_grid = np.arange(n) * dt
        if self.std == 0.0:
            self._x = np.zeros(n)
        else:
            rng = np.random.default_rng(self.seed)
            a = np.exp(-dt / self.tau_h)
            innov = self.std * np.sqrt(1.0 - a * a) * rng.standard_normal(n)
            x = np.empty(n)
            x[0] = self.std * rng.standard_normal()
            for k in range(1, n):
                x[k] = a * x[k - 1] + innov[k]
            self._x = x

    def __call__(self, t_now: float) -> float:
        base = self.base(t_now) if callable(self.base) else self.base
        return float(base + np.interp(t_now, self._t_grid, self._x))


class FeedDisturbance:

    def __init__(self, C_A0_gen=None, T_in_gen=None):
        self.C_A0_gen = C_A0_gen
        self.T_in_gen = T_in_gen
        feed = PARAMS["nominal_feed"]
        self.C_A0_nominal = feed["C_A0"]
        self.T_in_nominal = feed["T_in"]

    def __call__(self, t_now: float):
        C_A0 = self.C_A0_gen(t_now) if self.C_A0_gen is not None else self.C_A0_nominal
        T_in = self.T_in_gen(t_now) if self.T_in_gen is not None else self.T_in_nominal
        return C_A0, T_in


def step_disturbance(base: float, magnitude: float, t_start: float):
    return StepDisturbance(base, magnitude, t_start)


def ramp_disturbance(base: float, magnitude: float, t_start: float, t_end: float):
    return RampDisturbance(base, magnitude, t_start, t_end)


def ou_noise_disturbance(base, std: float, tau_h: float, seed: int,
                          t_max: float, grid_per_tau: int = 20):
    return OUNoiseDisturbance(base, std, tau_h, seed, t_max, grid_per_tau)


def build_feed_disturbance(C_A0_gen=None, T_in_gen=None):
    return FeedDisturbance(C_A0_gen, T_in_gen)
