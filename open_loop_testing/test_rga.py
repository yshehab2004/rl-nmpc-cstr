import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.rga import steady_state_gain, relative_gain_array

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_gain_matrix_is_2x2():
    G = steady_state_gain(18.83, 134.14, 1.0, **KW)
    assert G.shape == (2, 2)


def test_gain_matrix_matches_finite_difference_on_the_closed_form_solve():
    from open_loop_testing.steady_state_map import solve_steady_state
    F0, T_R0, beta = 18.83, 134.14, 1.0
    C_a0, C_b0, T_K0, Q_dot0 = solve_steady_state(F0, T_R0, beta, **KW)

    def T_R_for_Q_dot(F, Q_dot_target, T_R_lo=100.0, T_R_hi=150.0):
        for _ in range(80):
            mid = 0.5 * (T_R_lo + T_R_hi)
            _, _, _, Q_dot_mid = solve_steady_state(F, mid, beta, **KW)
            if Q_dot_mid < Q_dot_target:
                T_R_lo = mid
            else:
                T_R_hi = mid
        return 0.5 * (T_R_lo + T_R_hi)

    h = 1e-3
    T_R_plus = T_R_for_Q_dot(F0 + h, Q_dot0)
    T_R_minus = T_R_for_Q_dot(F0 - h, Q_dot0)
    _, C_b_plus, _, _ = solve_steady_state(F0 + h, T_R_plus, beta, **KW)
    _, C_b_minus, _, _ = solve_steady_state(F0 - h, T_R_minus, beta, **KW)

    dT_R_dF_expected = (T_R_plus - T_R_minus) / (2 * h)
    dC_b_dF_expected = (C_b_plus - C_b_minus) / (2 * h)

    G = steady_state_gain(F0, T_R0, beta, **KW)
    assert G[1, 0] == pytest.approx(dT_R_dF_expected, rel=1e-2)
    assert G[0, 0] == pytest.approx(dC_b_dF_expected, rel=1e-2)


def test_rga_rows_and_columns_sum_to_one():
    G = steady_state_gain(18.83, 134.14, 1.0, **KW)
    Lambda = relative_gain_array(G)
    assert np.allclose(Lambda.sum(axis=1), 1.0)
    assert np.allclose(Lambda.sum(axis=0), 1.0)


def test_rga_of_a_diagonal_system_is_the_identity():
    G = np.array([[3.0, 0.0], [0.0, 5.0]])
    Lambda = relative_gain_array(G)
    assert np.allclose(Lambda, np.eye(2))
