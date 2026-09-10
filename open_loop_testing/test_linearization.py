import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.linearization import ode_rhs, linearize

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


@pytest.mark.parametrize("F,T_R,beta", [
    (18.83, 134.14, 1.0), (20.0, 134.0, 0.6), (35.0, 139.0, 0.4),
])
def test_ode_rhs_is_zero_at_a_solved_steady_state(F, T_R, beta):
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, **KW)
    dxdt = ode_rhs(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), beta=beta, **KW)
    assert np.allclose(dxdt, 0.0, atol=1e-6)


def test_linearize_returns_4x4_A_and_4x2_B():
    F, T_R, beta = 18.83, 134.14, 1.0
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, **KW)
    A, B = linearize(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), beta=beta, **KW)
    assert A.shape == (4, 4)
    assert B.shape == (4, 2)


def test_B_has_no_direct_coupling_from_Q_dot_to_concentrations():
    F, T_R, beta = 18.83, 134.14, 1.0
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, **KW)
    A, B = linearize(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), beta=beta, **KW)
    assert B[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert B[1, 1] == pytest.approx(0.0, abs=1e-9)


def test_nominal_operating_point_is_open_loop_stable():
    F, T_R, beta = 18.83, 134.14, 1.0
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, **KW)
    A, B = linearize(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), beta=beta, **KW)
    eigvals = np.linalg.eigvals(A)
    assert np.all(eigvals.real < 0)
