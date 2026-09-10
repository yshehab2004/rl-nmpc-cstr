import pytest

from models.params import load_params
from open_loop_testing.boundary_gradient import gradient_at_F

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"], price_energy=ECON["price_energy"])


def test_dC_b_dF_matches_finite_difference_by_hand():
    F, T_R, beta = 35.0, 133.0, 1.0
    h = 1e-3
    from open_loop_testing.steady_state_map import solve_steady_state
    _, C_b_plus, _, _ = solve_steady_state(F + h, T_R, beta, **KW)
    _, C_b_minus, _, _ = solve_steady_state(F - h, T_R, beta, **KW)
    expected = (C_b_plus - C_b_minus) / (2 * h)

    result = gradient_at_F(F, T_R, beta, **KW, **PRICES)
    assert result["dC_b_dF"] == pytest.approx(expected, rel=1e-4)


def test_dPi_dF_sign_is_consistent_with_profit_still_rising_at_F35_beta1():
    result = gradient_at_F(35.0, 133.0, 1.0, **KW, **PRICES)
    assert result["dPi_dF"] > 0
