import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.feasible_region import classify, FEASIBLE
from open_loop_testing.optimum_trajectory import compute_profit, find_optimum, evaluate_point, compute_gap_table

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"], price_energy=ECON["price_energy"])


def test_compute_profit_matches_the_oracle_formula():
    F, C_b, Q_dot = 20.0, 0.9, -3000.0
    expected = PRICES["price_B"] * F * C_b - PRICES["price_A0"] * F * KW["C_A0"] - PRICES["price_energy"] * (-Q_dot)
    assert compute_profit(F, C_b, KW["C_A0"], Q_dot, **PRICES) == pytest.approx(expected)


def test_find_optimum_returns_a_feasible_node():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    result = find_optimum(F_grid, T_R_grid, beta=1.0, **KW, **PRICES)
    _, _, _, Q_dot = solve_steady_state(result["F_star"], result["T_R_star"], 1.0, **KW)
    assert -8500.0 <= Q_dot <= 0.0
    assert result["T_R_star"] <= 140.0


def test_find_optimum_matches_brute_force_argmax():
    F_grid = np.linspace(5.0, 35.0, 15)
    T_R_grid = np.linspace(100.0, 150.0, 21)
    beta = 0.6
    result = find_optimum(F_grid, T_R_grid, beta=beta, **KW, **PRICES)

    best_profit, best_F, best_T_R = -np.inf, None, None
    for F in F_grid:
        for T_R in T_R_grid:
            C_a, C_b, _, Q_dot = solve_steady_state(F, T_R, beta, **KW)
            code = classify(np.array([Q_dot]), np.array([T_R]))[0]
            if code != FEASIBLE:
                continue
            profit = compute_profit(F, C_b, KW["C_A0"], Q_dot, **PRICES)
            if profit > best_profit:
                best_profit, best_F, best_T_R = profit, F, T_R

    assert result["F_star"] == pytest.approx(best_F)
    assert result["T_R_star"] == pytest.approx(best_T_R)
    assert result["profit_star"] == pytest.approx(best_profit)


def test_find_optimum_raises_when_grid_is_entirely_infeasible():
    F_grid = np.array([35.0])
    T_R_grid = np.array([100.0])
    with pytest.raises(ValueError):
        find_optimum(F_grid, T_R_grid, beta=1.0, **KW, **PRICES)


def test_evaluate_point_flags_a_clearly_infeasible_point():
    result = evaluate_point(35.0, 100.0, beta=1.0, **KW, **PRICES)
    assert result["feasible"] is False


def test_evaluate_point_flags_a_clearly_feasible_point():
    result = evaluate_point(20.0, 134.0, beta=1.0, **KW, **PRICES)
    assert result["feasible"] is True


def test_gap_is_zero_at_beta_1_by_construction():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    table = compute_gap_table(F_grid, T_R_grid, [1.0, 0.6], **KW, **PRICES)
    row = table[0]
    assert row["beta"] == 1.0
    assert row["fixed_feasible"] is True
    assert row["gap_pct"] == pytest.approx(0.0, abs=1e-6)
    assert row["Pi_fixed"] == pytest.approx(row["Pi_star"])


def test_gap_table_marks_gap_pct_none_when_fixed_point_infeasible():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    table = compute_gap_table(F_grid, T_R_grid, [1.0, 0.05], **KW, **PRICES)
    row = table[1]
    assert row["fixed_feasible"] is False
    assert row["gap_pct"] is None
