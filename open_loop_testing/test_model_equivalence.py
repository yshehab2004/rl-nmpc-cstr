import numpy as np
import pytest

from models.cstr_model import build_model
from models.params import load_params
from open_loop_testing.linearization import ode_rhs
from open_loop_testing.steady_state_map import solve_steady_state

PARAMS = load_params()
C = PARAMS["certain_params"]


def _do_mpc_rhs(model, x, u, alpha, beta, C_A0, T_in):
    tvp = model.tvp(0)
    tvp["C_A0_feed"] = C_A0
    tvp["T_in_feed"] = T_in
    p = model.p(0)
    p["alpha"] = alpha
    p["beta"] = beta
    rhs = model._rhs_fun(x, u, np.zeros((0, 1)), tvp, p, np.zeros((0, 1)))
    return np.array(rhs).flatten()


SAMPLE_STATES = [
    (2.1409, 1.0917, 134.14, 128.96, 18.83, -4495.7, 1.0, 1.0, 5.1, 130.0),
    (3.5000, 0.8000, 120.00, 115.00, 30.00, -2000.0, 1.0, 0.8, 5.1, 130.0),
    (4.2000, 0.5000, 145.00, 138.00, 35.00, -8500.0, 1.0, 0.6, 5.7, 130.0),
    (1.8000, 1.2000, 100.00,  95.00,  5.00,     0.0, 1.0, 0.4, 4.5, 130.0),
    (4.8000, 0.3000, 150.00, 142.00, 20.00, -6000.0, 1.0, 0.2, 5.1, 125.0),
    (2.9000, 0.9500, 138.50, 132.00, 25.75, -7300.0, 1.0, 0.3, 4.9, 135.0),
]


@pytest.mark.parametrize("sample", SAMPLE_STATES)
def test_numpy_ode_matches_the_do_mpc_model(sample):
    C_a, C_b, T_R, T_K, F, Q_dot, alpha, beta, C_A0, T_in = sample
    model = build_model()
    x = np.array([C_a, C_b, T_R, T_K])
    u = np.array([F, Q_dot])

    expected = _do_mpc_rhs(model, x, u, alpha, beta, C_A0, T_in)
    actual = ode_rhs(x, u, alpha=alpha, beta=beta, C_A0=C_A0, T_in=T_in, C=C)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_numpy_ode_matches_do_mpc_over_a_random_sweep():
    rng = np.random.default_rng(20260905)
    model = build_model()
    for _ in range(200):
        C_a = rng.uniform(0.5, 5.5)
        C_b = rng.uniform(0.1, 1.5)
        T_R = rng.uniform(100.0, 150.0)
        T_K = rng.uniform(90.0, 150.0)
        F = rng.uniform(5.0, 35.0)
        Q_dot = rng.uniform(-8500.0, 0.0)
        beta = rng.uniform(0.15, 1.0)
        C_A0 = rng.uniform(4.5, 5.7)
        T_in = rng.uniform(125.0, 135.0)

        x = np.array([C_a, C_b, T_R, T_K])
        u = np.array([F, Q_dot])
        expected = _do_mpc_rhs(model, x, u, 1.0, beta, C_A0, T_in)
        actual = ode_rhs(x, u, alpha=1.0, beta=beta, C_A0=C_A0, T_in=T_in, C=C)

        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("F,T_R,beta", [
    (18.83, 134.14, 1.0), (35.0, 133.0, 1.0), (35.0, 135.5, 0.8),
    (35.0, 140.0, 0.6), (25.91, 140.0, 0.3), (18.35, 140.0, 0.2),
])
def test_closed_form_steady_state_is_a_steady_state_of_the_do_mpc_model(F, T_R, beta):
    model = build_model()
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, 1.0, 5.1, 130.0, C)
    x = np.array([float(C_a), float(C_b), T_R, float(T_K)])
    u = np.array([F, float(Q_dot)])

    dxdt = _do_mpc_rhs(model, x, u, 1.0, beta, 5.1, 130.0)
    assert dxdt == pytest.approx(np.zeros(4), abs=1e-9)


def test_controller_model_matches_the_plant_model_when_beliefs_are_correct():
    plant = build_model(for_controller=False)
    controller = build_model(for_controller=True)
    x = np.array([2.1409, 1.0917, 134.14, 128.96])
    u = np.array([18.83, -4495.7])
    beta = 0.55

    plant_rhs = _do_mpc_rhs(plant, x, u, 1.0, beta, 5.1, 130.0)

    tvp = controller.tvp(0)
    tvp["C_A0_feed"] = 5.1
    tvp["T_in_feed"] = 130.0
    tvp["beta_belief"] = beta
    p = controller.p(0)
    p["alpha"] = 1.0
    p["beta"] = 0.0
    controller_rhs = np.array(
        controller._rhs_fun(x, u, np.zeros((0, 1)), tvp, p, np.zeros((0, 1)))
    ).flatten()

    assert controller_rhs == pytest.approx(plant_rhs, rel=1e-12, abs=1e-12)
