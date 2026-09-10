import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.optimum_trajectory import compute_profit, find_optimum
from open_loop_testing.profit_contours import find_unconstrained_optimum

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"], price_energy=ECON["price_energy"])


def test_unconstrained_optimum_matches_brute_force():
    F_grid = np.linspace(5.0, 35.0, 15)
    T_R_grid = np.linspace(100.0, 150.0, 21)
    result = find_unconstrained_optimum(F_grid, T_R_grid, beta=0.6, **KW, **PRICES)

    from open_loop_testing.steady_state_map import solve_steady_state
    best_profit, best_F, best_T_R = -np.inf, None, None
    for F in F_grid:
        for T_R in T_R_grid:
            _, C_b, _, Q_dot = solve_steady_state(F, T_R, 0.6, **KW)
            profit = compute_profit(F, C_b, KW["C_A0"], Q_dot, **PRICES)
            if profit > best_profit:
                best_profit, best_F, best_T_R = profit, F, T_R
    assert result["F_star"] == pytest.approx(best_F)
    assert result["T_R_star"] == pytest.approx(best_T_R)


def test_unconstrained_optimum_has_at_least_as_much_profit_as_constrained():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    unconstrained = find_unconstrained_optimum(F_grid, T_R_grid, beta=0.4, **KW, **PRICES)
    constrained = find_optimum(F_grid, T_R_grid, beta=0.4, **KW, **PRICES)
    assert unconstrained["profit_star"] >= constrained["profit_star"] - 1e-9
