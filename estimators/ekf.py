import numpy as np
from scipy.linalg import expm

from models.params import load_params
from open_loop_testing.linearization import ode_rhs
from simulation.measurement import measurement_covariance

PARAMS = load_params()
_EST = PARAMS["estimator"]
_CERTAIN = PARAMS["certain_params"]

N_PHYS = 4
N_AUG = 5
BETA_INDEX = 4

H = np.zeros((2, N_AUG))
H[0, 2] = 1.0
H[1, 1] = 1.0


class BetaEKF:

    def __init__(self, x0, beta0=1.0, alpha=None, C=None,
                 q_states=None, q_beta=None, p0_states=None, p0_beta=None,
                 noise_frac=None, beta_bounds=None):
        alpha = PARAMS["uncertain_params"]["alpha_nominal"] if alpha is None else alpha
        self.alpha = alpha
        self.C = _CERTAIN if C is None else C

        q_states = _EST["q_states"] if q_states is None else q_states
        q_beta = _EST["q_beta"] if q_beta is None else q_beta
        p0_states = _EST["p0_states"] if p0_states is None else p0_states
        p0_beta = _EST["p0_beta"] if p0_beta is None else p0_beta
        self.beta_bounds = ((_EST["beta_min"], _EST["beta_max"])
                             if beta_bounds is None else beta_bounds)

        self.x = np.concatenate([np.asarray(x0, dtype=float).flatten()[:N_PHYS],
                                  [beta0]])
        self.P = np.diag([p0_states] * N_PHYS + [p0_beta])
        self.Q = np.diag([q_states] * N_PHYS + [q_beta])
        self.R = measurement_covariance(noise_frac)


    def _f(self, x_aug, u, C_A0, T_in):
        dx = ode_rhs(x_aug[:N_PHYS], u, alpha=self.alpha, beta=x_aug[BETA_INDEX],
                      C_A0=C_A0, T_in=T_in, C=self.C)
        return np.concatenate([dx, [0.0]])

    def _jacobian(self, x_aug, u, C_A0, T_in, eps=1e-6):
        A = np.zeros((N_AUG, N_AUG))
        for i in range(N_AUG):
            h = eps * max(abs(x_aug[i]), 1.0)
            xp, xm = x_aug.copy(), x_aug.copy()
            xp[i] += h
            xm[i] -= h
            A[:, i] = (self._f(xp, u, C_A0, T_in) - self._f(xm, u, C_A0, T_in)) / (2 * h)
        return A


    def predict(self, u, dt, C_A0, T_in):
        u = np.asarray(u, dtype=float).flatten()
        f = lambda x: self._f(x, u, C_A0, T_in)

        k1 = f(self.x)
        k2 = f(self.x + 0.5 * dt * k1)
        k3 = f(self.x + 0.5 * dt * k2)
        k4 = f(self.x + dt * k3)
        self.x = self.x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        A = self._jacobian(self.x, u, C_A0, T_in)
        F = expm(A * dt)
        self.P = F @ self.P @ F.T + self.Q * dt
        self._clamp_beta()
        return self.x.copy()

    def update(self, y):
        y = np.asarray(y, dtype=float).flatten()
        innovation = y - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        IKH = np.eye(N_AUG) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self._clamp_beta()
        return self.x.copy()

    def _clamp_beta(self):
        lo, hi = self.beta_bounds
        self.x[BETA_INDEX] = float(np.clip(self.x[BETA_INDEX], lo, hi))


    @property
    def beta_hat(self):
        return float(self.x[BETA_INDEX])

    @property
    def x_hat(self):
        return self.x[:N_PHYS].copy()
