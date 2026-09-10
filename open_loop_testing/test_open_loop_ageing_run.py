import numpy as np
import pytest

from models.params import load_params
from models.deactivation import make_severity_fn
from open_loop_testing.open_loop_ageing_run import simulate_open_loop_ageing

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(C_A0=5.1, T_in=130.0, C=C)
CAMPAIGN_H = 75.0


def test_beta_in_the_simulation_matches_the_schedule_at_the_endpoints():
    schedule = make_severity_fn("severe", CAMPAIGN_H, profile="exponential")
    F0, T_R0 = 35.0, 133.0
    from open_loop_testing.steady_state_map import solve_steady_state
    C_a0, C_b0, T_K0, Q_dot0 = solve_steady_state(F0, T_R0, 1.0, alpha=1.0, **KW)

    result = simulate_open_loop_ageing(np.array([C_a0, C_b0, T_R0, T_K0]), np.array([F0, Q_dot0]),
                                          schedule, CAMPAIGN_H, alpha=1.0, **KW, n=50)
    assert result["beta"][0] == pytest.approx(1.0, abs=1e-6)
    assert result["beta"][-1] == pytest.approx(0.2, abs=1e-6)


def test_T_R_drifts_up_when_fixed_Q_dot_can_no_longer_hold_it():
    schedule = make_severity_fn("severe", CAMPAIGN_H, profile="exponential")
    F0, T_R0 = 35.0, 133.0
    from open_loop_testing.steady_state_map import solve_steady_state
    C_a0, C_b0, T_K0, Q_dot0 = solve_steady_state(F0, T_R0, 1.0, alpha=1.0, **KW)

    result = simulate_open_loop_ageing(np.array([C_a0, C_b0, T_R0, T_K0]), np.array([F0, Q_dot0]),
                                          schedule, CAMPAIGN_H, alpha=1.0, **KW, n=200)
    T_R_traj = result["x"][:, 2]
    assert T_R_traj[-1] > T_R_traj[0] + 1.0
    assert np.all(np.diff(T_R_traj) > -1e-6)
