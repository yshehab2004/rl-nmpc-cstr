import numpy as np
import pytest

from models.params import load_params
from models.deactivation import make_severity_fn
from open_loop_testing.feasible_region import classify, FEASIBLE
from open_loop_testing.campaign_integrated_gap import (
    find_best_always_feasible_point, campaign_total_profit_fixed,
    campaign_total_profit_adaptive,
)

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"], price_energy=ECON["price_energy"])
CAMPAIGN_H = 75.0


def test_feasible_at_worst_beta_is_also_feasible_at_mild_beta_spot_check():
    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid, betas, Q_dot = d["F"], d["T_R"], d["beta"], d["Q_dot"]
    i02 = list(betas).index(0.2)
    i10 = list(betas).index(1.0)
    T_R_mesh = np.broadcast_to(T_R_grid[None, :], Q_dot[0].shape)
    feas_02 = classify(Q_dot[i02], T_R_mesh) == FEASIBLE
    feas_10 = classify(Q_dot[i10], T_R_mesh) == FEASIBLE
    assert np.all(feas_10[feas_02])


def test_best_always_feasible_point_is_feasible_at_the_worst_beta():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    schedule = make_severity_fn("severe", CAMPAIGN_H, profile="exponential")
    result = find_best_always_feasible_point(F_grid, T_R_grid, schedule, CAMPAIGN_H, **KW, **PRICES)
    _, _, _, Q_dot = __import__("open_loop_testing.steady_state_map", fromlist=["solve_steady_state"]).solve_steady_state(
        result["F"], result["T_R"], 0.2, **KW)
    assert -8500.0 <= Q_dot <= 0.0
    assert result["T_R"] <= 140.0


def test_adaptive_campaign_profit_is_at_least_the_fixed_points():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    schedule = make_severity_fn("severe", CAMPAIGN_H, profile="exponential")

    adaptive_total = campaign_total_profit_adaptive(F_grid, T_R_grid, schedule, CAMPAIGN_H, **KW, **PRICES)
    fixed_total = campaign_total_profit_fixed(10.0, 135.0, schedule, CAMPAIGN_H, **KW, **PRICES)
    assert not np.isnan(fixed_total)
    assert adaptive_total >= fixed_total
