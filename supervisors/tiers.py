import numpy as np

from models.params import load_params
from open_loop_testing.optimum_trajectory import find_optimum

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]
ECON = PARAMS["economics"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
ALPHA = PARAMS["uncertain_params"]["alpha_nominal"]

_MARGIN_BRACKETS = [(0.6, 0.0), (0.4, 2.0), (0.3, 2.25)]
_MARGIN_FLOOR = 2.5

BETA_HAT_UNCERTAINTY = 0.04


def margin_for_beta(beta):
    for threshold, margin in _MARGIN_BRACKETS:
        if beta >= threshold:
            return margin
    return _MARGIN_FLOOR


class FixedSupervisor:

    def __init__(self, F_ref, T_R_ref, margin, name):
        self.target = (float(F_ref), float(T_R_ref), float(margin))
        self.name = name
        self.n_solves = 0

    def get_target(self, t_now, beta_hat=None, C_A0=None, T_in=None,
                    x_hat=None, u_last=None):
        return self.target


def T0a_naive():
    return FixedSupervisor(35.0, 133.0, 0.0, name="T0a_naive")


def T0b_prudent(margin=2.5):
    return FixedSupervisor(18.25, 140.0, margin, name="T0b_prudent")


class RTOSupervisor:

    def __init__(self, cadence_h=2.0, margin=None, uncertainty=BETA_HAT_UNCERTAINTY,
                 F_grid=None, T_R_grid=None, name="T2_rto"):
        self.cadence_h = float(cadence_h)
        self.fixed_margin = margin
        self.uncertainty = float(uncertainty)
        self.name = name
        self.F_grid = np.linspace(5.0, 35.0, 601) if F_grid is None else F_grid
        self.T_R_grid = np.linspace(100.0, 150.0, 501) if T_R_grid is None else T_R_grid

        self._last_solve_t = None
        self._target = None
        self.n_solves = 0
        self.history = []

    def _solve(self, beta_eval, margin, C_A0, T_in):
        return find_optimum(
            self.F_grid, self.T_R_grid, beta_eval, ALPHA, C_A0, T_in, C,
            **PRICES, T_R_max=T_LIMIT - margin,
        )

    def get_target(self, t_now, beta_hat=None, C_A0=None, T_in=None,
                    x_hat=None, u_last=None):
        beta_hat = 1.0 if beta_hat is None else float(beta_hat)
        C_A0 = FEED["C_A0"] if C_A0 is None else float(C_A0)
        T_in = FEED["T_in"] if T_in is None else float(T_in)

        due = (self._last_solve_t is None
               or (t_now - self._last_solve_t) >= self.cadence_h - 1e-9)
        if not due:
            return self._target

        beta_eval = float(np.clip(beta_hat - self.uncertainty, 0.05, 1.0))
        margin = self.fixed_margin if self.fixed_margin is not None else margin_for_beta(beta_eval)

        try:
            opt = self._solve(beta_eval, margin, C_A0, T_in)
            self._target = (opt["F_star"], opt["T_R_star"], margin)
        except ValueError:
            if self._target is None:
                raise
        self._last_solve_t = t_now
        self.n_solves += 1
        self.history.append({"t": t_now, "beta_hat": beta_hat, "beta_eval": beta_eval,
                              "margin": margin, "target": self._target})
        return self._target
