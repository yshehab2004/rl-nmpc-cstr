import numpy as np
import pytest

from controllers.nmpc import build_nmpc, run_closed_loop
from models.params import load_params
from simulation.simulator import build_simulator, initial_state

PARAMS = load_params()
C = PARAMS["certain_params"]
Q_LOWER, Q_UPPER = -8500.0, 0.0
T_R_LIMIT = 140.0

SOLVER_TOL = 1e-3

FEASIBLE_TARGET = (18.83, 134.14)


@pytest.fixture(scope="module")
def nominal_run():
    mpc = build_nmpc(target=FEASIBLE_TARGET, n_horizon=50)
    simulator = build_simulator()
    return run_closed_loop(mpc, simulator, initial_state(), n_steps=50)


def test_tracks_a_feasible_fixed_target_over_50_steps(nominal_run):
    T_R = nominal_run["x"][:, 2]
    assert np.all(np.isfinite(nominal_run["x"]))
    assert np.all(np.isfinite(nominal_run["u"]))
    assert T_R.min() > 50.0 and T_R.max() < 200.0
    assert T_R[-1] == pytest.approx(FEASIBLE_TARGET[1], abs=1.0)


def test_settles_rather_than_oscillating(nominal_run):
    F = nominal_run["u"][:, 0]
    first_half_tv = np.abs(np.diff(F[: len(F) // 2])).sum()
    second_half_tv = np.abs(np.diff(F[len(F) // 2 :])).sum()
    assert second_half_tv < first_half_tv


def test_respects_the_hard_cooling_bound(nominal_run):
    Q_dot = nominal_run["u"][:, 1]
    assert Q_dot.min() >= Q_LOWER - SOLVER_TOL
    assert Q_dot.max() <= Q_UPPER + SOLVER_TOL


def test_respects_the_feed_rate_bounds(nominal_run):
    F = nominal_run["u"][:, 0]
    assert F.min() >= PARAMS["input_bounds"]["F"]["lower"] - SOLVER_TOL
    assert F.max() <= PARAMS["input_bounds"]["F"]["upper"] + SOLVER_TOL


def test_temperature_limit_is_respected_at_a_feasible_target(nominal_run):
    T_R = nominal_run["x"][:, 2]
    assert T_R.max() <= T_R_LIMIT + 0.05


def test_tracks_the_sstos_own_output():
    from open_loop_testing.target_optimizer import solve_target

    projected = solve_target(35.0, 133.0, margin=0.0, beta_hat=1.0, C_A0=5.1,
                              alpha=1.0, T_in=130.0, C=C)
    mpc = build_nmpc(target=(projected["F_s"], projected["T_R_s"]), n_horizon=50)
    simulator = build_simulator()
    run = run_closed_loop(mpc, simulator, initial_state(), n_steps=50)

    assert np.all(np.isfinite(run["x"]))
    assert run["x"][:, 2].max() <= T_R_LIMIT + 0.05
    assert run["u"][:, 1].min() >= Q_LOWER - SOLVER_TOL


def test_holds_target_under_ageing_without_adapting():
    from models.deactivation import make_deactivation_fn

    ageing_fn = make_deactivation_fn(beta_final=0.6, campaign_h=1.0)
    mpc = build_nmpc(target=FEASIBLE_TARGET, n_horizon=50)
    simulator = build_simulator(ageing_fn=ageing_fn)
    run = run_closed_loop(mpc, simulator, initial_state(), n_steps=100)

    assert np.all(np.isfinite(run["x"]))
    assert run["u"][:, 1].min() >= Q_LOWER - SOLVER_TOL
    assert run["u"][:, 1].max() <= Q_UPPER + SOLVER_TOL


def test_does_not_chatter_at_severe_ageing_under_an_unmeasured_disturbance():
    from models.disturbances import build_feed_disturbance, step_disturbance
    from open_loop_testing.target_optimizer import solve_target

    feed = PARAMS["nominal_feed"]
    projected = solve_target(35.0, 133.0, margin=0.0, beta_hat=0.2,
                              C_A0=feed["C_A0"], alpha=1.0, T_in=feed["T_in"], C=C)
    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(base=feed["C_A0"], magnitude=0.6, t_start=0.1)
    )

    mpc = build_nmpc(target=(projected["F_s"], projected["T_R_s"]),
                      n_horizon=20, beta_belief=0.2)
    simulator = build_simulator(ageing_fn=lambda t_now: (1.0, 0.2),
                                 disturbance_fn=disturbed)
    run = run_closed_loop(mpc, simulator, initial_state(), n_steps=100)

    F = run["u"][:, 0]
    assert np.abs(np.diff(F)).sum() < 50.0

    dF = np.diff(F[22:])
    sign_changes = int(np.sum(np.sign(dF[1:]) * np.sign(dF[:-1]) < 0))
    assert sign_changes < 0.25 * (len(dF) - 1)
    assert np.abs(dF).max() < 2.0

    assert run["u"][:, 1].min() >= Q_LOWER - SOLVER_TOL


def test_controller_belief_is_nominal_and_does_not_track_the_plant():
    mpc = build_nmpc(target=FEASIBLE_TARGET, n_horizon=50)
    beta_nominal = PARAMS["uncertain_params"]["beta_nominal"]
    for t_now in (0.0, 0.05, 0.5):
        tvp = mpc.tvp_fun(t_now)
        assert float(tvp["_tvp", 0, "beta_belief"]) == pytest.approx(beta_nominal)
