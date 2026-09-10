import numpy as np
import pytest

from estimators.run_estimator import run
from models.params import load_params
from open_loop_testing.steady_state_map import solve_steady_state

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]

F0, T_R0, BETA0 = 18.83, 134.14, 1.0
_C_a, _C_b, _T_K, _Q_dot = solve_steady_state(F0, T_R0, BETA0, 1.0,
                                                FEED["C_A0"], FEED["T_in"], C)
X_SS = np.array([float(_C_a), float(_C_b), T_R0, float(_T_K)])
U_SS = np.array([F0, float(_Q_dot)])


def test_converges_to_true_states_at_nominal_without_noise():
    out = run(X_SS, U_SS, n_steps=100, noise=False, beta0=1.0)
    err = np.abs(out["x_hat"][-1] - out["x_true"][-1])
    assert err[0] < 1e-3
    assert err[1] < 1e-4
    assert err[2] < 1e-3
    assert err[3] < 1e-2
    assert out["beta_hat"][-1] == pytest.approx(1.0, abs=0.02)


def test_recovers_beta_from_a_wrong_initial_guess():
    out = run(X_SS, U_SS, n_steps=400, noise=False, beta0=1.0,
               ageing_fn=lambda t_now: (1.0, 0.6))
    assert abs(out["beta_hat"][-1] - 0.6) < abs(1.0 - 0.6) * 0.5
    assert out["beta_hat"][-1] < out["beta_hat"][0]


def test_beta_hat_tracks_decaying_beta_with_lag():
    from models.deactivation import make_deactivation_fn

    ageing_fn = make_deactivation_fn(beta_final=0.5, campaign_h=2.0)
    out = run(X_SS, U_SS, n_steps=400, noise=False, beta0=1.0, ageing_fn=ageing_fn)

    final_err = abs(out["beta_hat"][-1] - out["beta_true"][-1])
    frozen_err = abs(1.0 - out["beta_true"][-1])
    assert final_err < frozen_err

    assert out["beta_hat"][-1] >= out["beta_true"][-1] - 0.05
    assert np.all(np.diff(out["beta_true"]) <= 1e-9)


def test_tracks_the_real_campaign_rate_closely_with_noise_on():
    from models.deactivation import make_deactivation_fn

    ageing_fn = make_deactivation_fn(beta_final=0.2, campaign_h=75.0)
    out = run(X_SS, U_SS, n_steps=3000, noise=True, seed=11,
               beta0=1.0, ageing_fn=ageing_fn)
    lag = np.abs(out["beta_true"] - out["beta_hat"])
    assert lag.max() < 0.06
    assert lag.mean() < 0.03


def test_estimates_stay_bounded_with_noise_on():
    out = run(X_SS, U_SS, n_steps=400, noise=True, seed=20260905, beta0=1.0)
    assert np.all(np.isfinite(out["x_hat"]))
    assert np.all(np.isfinite(out["beta_hat"]))
    err = np.abs(out["x_hat"] - out["x_true"])
    assert err[:, 2].max() < 2.0
    assert err[:, 1].max() < 0.1
    assert 0.05 <= out["beta_hat"].min() and out["beta_hat"].max() <= 1.0


def test_beta_stays_within_physical_bounds():
    from models.disturbances import build_feed_disturbance, step_disturbance

    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(base=FEED["C_A0"], magnitude=0.6, t_start=0.25)
    )
    out = run(X_SS, U_SS, n_steps=300, noise=True, seed=7,
               disturbance_fn=disturbed, C_A0_known=True)
    assert out["beta_hat"].min() >= 0.05 - 1e-9
    assert out["beta_hat"].max() <= 1.0 + 1e-9


def test_measured_C_A0_prevents_the_feed_step_aliasing_into_beta():
    from models.disturbances import build_feed_disturbance, step_disturbance

    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(base=FEED["C_A0"], magnitude=0.6, t_start=0.25)
    )
    kw = dict(n_steps=300, noise=False, beta0=1.0, disturbance_fn=disturbed)

    known = run(X_SS, U_SS, C_A0_known=True, **kw)
    blind = run(X_SS, U_SS, C_A0_known=False, **kw)

    drift_known = abs(known["beta_hat"][-1] - 1.0)
    drift_blind = abs(blind["beta_hat"][-1] - 1.0)

    assert drift_known < drift_blind
    assert drift_known < 0.05
