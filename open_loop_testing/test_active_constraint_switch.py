import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.active_constraint_switch import cooling_floor_T_R, peak_cooling_floor_T_R, find_switch_beta

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_cooling_floor_T_R_matches_coarse_grid_boundary():
    assert cooling_floor_T_R(35.0, 0.60, **KW) == pytest.approx(136.34, abs=0.01)
    assert cooling_floor_T_R(35.0, 0.40, **KW) == pytest.approx(139.67, abs=0.01)
    assert cooling_floor_T_R(35.0, 0.20, **KW) == pytest.approx(146.13, abs=0.01)


def test_cooling_floor_T_R_returns_lower_bound_when_already_feasible_there():
    result = cooling_floor_T_R(5.0, 1.0, T_R_lo=100.0, T_R_hi=150.0, **KW)
    assert result == 100.0


def test_peak_cooling_floor_T_R_occurs_at_the_highest_F():
    peak, F_at_peak = peak_cooling_floor_T_R(0.30, F_range=(5.0, 35.0), **KW)
    assert F_at_peak == pytest.approx(35.0, abs=0.3)


@pytest.mark.parametrize("beta,expected_peak", [(0.60, 136.34), (0.40, 139.67), (0.20, 146.13)])
def test_peak_cooling_floor_T_R_matches_hand_computed_values(beta, expected_peak):
    peak, _ = peak_cooling_floor_T_R(beta, F_range=(5.0, 35.0), **KW)
    assert peak == pytest.approx(expected_peak, abs=0.01)


def test_find_switch_beta_is_between_035_and_040():
    beta_star = find_switch_beta(beta_lo=0.35, beta_hi=0.40, T_R_max=140.0, **KW)
    assert 0.35 < beta_star < 0.40

    peak_at_star, _ = peak_cooling_floor_T_R(beta_star, F_range=(5.0, 35.0), **KW)
    assert peak_at_star == pytest.approx(140.0, abs=1e-4)
