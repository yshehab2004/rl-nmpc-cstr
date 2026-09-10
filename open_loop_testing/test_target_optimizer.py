import pytest

from models.params import load_params
from open_loop_testing.target_optimizer import solve_target

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, T_in=130.0, C=C)


def test_feasible_target_passes_through_unchanged():
    result = solve_target(35.0, 133.0, margin=0.0, beta_hat=1.0, C_A0=5.1, **KW)
    assert result["F_s"] == pytest.approx(35.0, abs=1e-6)
    assert result["T_R_s"] == pytest.approx(133.0, abs=1e-6)


def test_infeasible_target_returns_a_feasible_point():
    result = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.8, C_A0=5.1, **KW)
    from open_loop_testing.steady_state_map import solve_steady_state
    _, _, _, Q_dot = solve_steady_state(result["F_s"], result["T_R_s"], 0.8,
                                          alpha=KW["alpha"], C_A0=5.1, T_in=KW["T_in"], C=C)
    assert -8500.0 <= Q_dot <= 0.0
    assert result["T_R_s"] <= 140.0


def test_output_is_a_genuine_steady_state():
    import numpy as np

    from open_loop_testing.linearization import ode_rhs
    from open_loop_testing.steady_state_map import solve_steady_state

    result = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.8, C_A0=5.1, **KW)
    C_a, C_b, T_K, Q_dot = solve_steady_state(result["F_s"], result["T_R_s"], 0.8,
                                                alpha=KW["alpha"], C_A0=5.1, T_in=KW["T_in"], C=C)
    x = np.array([C_a, C_b, result["T_R_s"], T_K])
    u = np.array([result["F_s"], Q_dot])
    dxdt = ode_rhs(x, u, alpha=KW["alpha"], beta=0.8, C_A0=5.1, T_in=KW["T_in"], C=C)
    assert dxdt == pytest.approx(np.zeros(4), abs=1e-6)


def test_C_A0_changes_the_projection():
    result_45 = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.8, C_A0=4.5, **KW)
    result_57 = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.8, C_A0=5.7, **KW)
    assert (result_45["F_s"], result_45["T_R_s"]) != pytest.approx(
        (result_57["F_s"], result_57["T_R_s"]))


def test_beta_hat_changes_the_projection():
    result_06 = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.6, C_A0=5.1, **KW)
    result_02 = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.2, C_A0=5.1, **KW)
    assert (result_06["F_s"], result_06["T_R_s"]) != pytest.approx(
        (result_02["F_s"], result_02["T_R_s"]))


def test_margin_tightens_the_temperature_ceiling():
    unmargined = solve_target(20.0, 139.0, margin=0.0, beta_hat=1.0, C_A0=5.1, **KW)
    assert unmargined["T_R_s"] == pytest.approx(139.0, abs=1e-6)

    margined = solve_target(20.0, 139.0, margin=5.0, beta_hat=1.0, C_A0=5.1, **KW)
    assert margined["T_R_s"] <= 135.0 + 1e-6


def test_margin_backs_off_the_cooling_floor():
    from open_loop_testing.active_constraint_multipliers import _T_R_for_cooling_floor

    margin = 3.0
    result = solve_target(35.0, 133.0, margin=margin, beta_hat=0.8, C_A0=5.1, **KW)
    floor = _T_R_for_cooling_floor(result["F_s"], 0.8, alpha=KW["alpha"], C_A0=5.1, T_in=KW["T_in"], C=C)
    assert result["T_R_s"] >= floor + margin - 1e-3


def test_metric_is_geometric_range_normalized_not_economic():
    from open_loop_testing.active_constraint_multipliers import _T_R_for_cooling_floor
    from open_loop_testing.target_optimizer import F_BOUNDS_DEFAULT, T_R_BOUNDS_DEFAULT

    result = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.8, C_A0=5.1, **KW)
    F_s, T_R_s = result["F_s"], result["T_R_s"]

    h = 1e-3
    floor_prime = (_T_R_for_cooling_floor(F_s + h, 0.8, alpha=KW["alpha"], C_A0=5.1, T_in=KW["T_in"], C=C)
                   - _T_R_for_cooling_floor(F_s - h, 0.8, alpha=KW["alpha"], C_A0=5.1, T_in=KW["T_in"], C=C)) / (2 * h)

    F_range = F_BOUNDS_DEFAULT[1] - F_BOUNDS_DEFAULT[0]
    T_R_range = T_R_BOUNDS_DEFAULT[1] - T_R_BOUNDS_DEFAULT[0]
    w_ratio = (F_range / T_R_range) ** 2

    predicted_F_shift = -w_ratio * floor_prime * (T_R_s - 133.0)
    assert (F_s - 35.0) == pytest.approx(predicted_F_shift, abs=1e-2)


def test_projection_is_idempotent():
    once = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.8, C_A0=5.1, **KW)
    twice = solve_target(once["F_s"], once["T_R_s"], margin=0.0, beta_hat=0.8, C_A0=5.1, **KW)
    assert twice["F_s"] == pytest.approx(once["F_s"], abs=1e-6)
    assert twice["T_R_s"] == pytest.approx(once["T_R_s"], abs=1e-6)
    assert twice["projected"] is False
