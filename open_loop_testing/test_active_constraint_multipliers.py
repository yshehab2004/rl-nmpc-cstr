import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.active_constraint_multipliers import identify_active_constraints, compute_multipliers

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"], price_energy=ECON["price_energy"])


def test_identifies_F_bound_and_cooling_floor_at_beta_1():
    active = identify_active_constraints(35.0, 133.0, 1.0, T_lim=140.0, **KW)
    assert "F_upper" in active
    assert "Q_dot_lower" in active
    assert "T_R_upper" not in active


def test_identifies_F_bound_and_T_R_bound_at_beta_0_6():
    active = identify_active_constraints(35.0, 140.0, 0.6, T_lim=140.0, **KW)
    assert "T_R_upper" in active
    assert "F_upper" in active
    assert "Q_dot_lower" not in active


def test_multipliers_are_all_non_negative():
    result = compute_multipliers(35.0, 133.0, 1.0, T_lim=140.0, **KW, **PRICES)
    for name, lam in result["multipliers"].items():
        assert lam >= -1e-6, f"{name} multiplier was negative: {lam}"


def test_vertex_reconstruction_satisfies_the_cooling_floor_exactly():
    from open_loop_testing.active_constraint_multipliers import _vertex_point
    from open_loop_testing.steady_state_map import solve_steady_state

    F_v, T_R_v = _vertex_point({"F_upper", "Q_dot_lower"}, 35.0, 140.0, 1.0, **KW)
    _, _, _, Q_dot = solve_steady_state(F_v, T_R_v, 1.0, **KW)
    assert Q_dot == pytest.approx(-8500.0, abs=1e-4)

    F_v2, T_R_v2 = _vertex_point({"T_R_upper", "Q_dot_lower"}, 35.0, 140.0, 0.2, **KW)
    assert T_R_v2 == 140.0
    _, _, _, Q_dot2 = solve_steady_state(F_v2, T_R_v2, 0.2, **KW)
    assert Q_dot2 == pytest.approx(-8500.0, abs=1e-4)


def test_beta_0_8_is_a_genuine_edge_with_real_slack_not_a_vertex():
    from open_loop_testing.steady_state_map import solve_steady_state
    _, _, _, Q_dot = solve_steady_state(35.0, 135.5, 0.8, **KW)
    assert Q_dot - (-8500.0) > 500.0

    result = compute_multipliers(35.0, 135.5, 0.8, T_lim=140.0, **KW, **PRICES)
    assert result["active_constraints"] == ["F_upper"]
    assert "reduced_d2_profit_dT_R2" in result
    assert result["reduced_d2_profit_dT_R2"] < 0
    assert result["multipliers"]["F_upper"] > 0


def test_F_for_cooling_floor_agrees_with_cooling_floor_T_R_the_other_way():
    from open_loop_testing.active_constraint_multipliers import _F_for_cooling_floor
    from open_loop_testing.active_constraint_switch import cooling_floor_T_R

    T_R_known = cooling_floor_T_R(20.0, 0.6, **KW)
    F_recovered = _F_for_cooling_floor(T_R_known, 0.6, **KW)
    assert F_recovered == pytest.approx(20.0, abs=1e-3)
