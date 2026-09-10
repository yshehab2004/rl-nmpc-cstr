import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.nmp_zero import transmission_zeros, F_to_Cb_zeros, simulate_step_response

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_transmission_zeros_of_a_known_synthetic_system():
    A = np.array([[0.0, 1.0], [-3.0, -4.0]])
    B = np.array([[0.0], [1.0]])
    Cc = np.array([-2.0, 1.0])
    zeros = transmission_zeros(A, B[:, 0], Cc)
    assert len(zeros) == 1
    assert zeros[0] == pytest.approx(2.0, abs=1e-6)


def test_F_to_Cb_has_a_positive_real_zero_at_the_nominal_point():
    zeros = F_to_Cb_zeros(18.83, 134.14, 1.0, **KW)
    assert any(z.real > 0 for z in zeros)


def test_rhp_zero_prediction_matches_an_actual_inverse_response_simulation():
    from open_loop_testing.steady_state_map import solve_steady_state

    F0, T_R0, beta = 18.83, 134.14, 1.0
    C_a0, C_b0, T_K0, Q_dot0 = solve_steady_state(F0, T_R0, beta, **KW)
    x0 = np.array([C_a0, C_b0, T_R0, T_K0])
    u0 = np.array([F0, Q_dot0])

    dF = 0.5
    t, x_traj = simulate_step_response(x0, u0, du=np.array([dF, 0.0]), beta=beta, dt=1e-4,
                                         n_steps=2000, **KW)
    C_b_traj = x_traj[:, 1]

    C_a_new, C_b_new, _, _ = solve_steady_state(F0 + dF, T_R0, beta, **KW)
    initial_direction = np.sign(C_b_traj[5] - C_b0)
    final_direction = np.sign(C_b_new - C_b0)

    zeros = F_to_Cb_zeros(F0, T_R0, beta, **KW)
    predicted_inverse_response = any(z.real > 0 for z in zeros)

    if predicted_inverse_response:
        assert initial_direction != final_direction
