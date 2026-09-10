import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.steady_state_map import rate_constants, solve_steady_state, build_map

PARAMS = load_params()
C = PARAMS["certain_params"]
THETA = 134.14


def test_rate_constants_at_nominal_point():
    k1, k2, k3 = rate_constants(THETA, alpha=1.0, beta=1.0, C=C)
    assert k1 == pytest.approx(50.61, rel=1e-3)
    assert k2 == pytest.approx(50.61, rel=1e-3)
    assert k1 == k2


def test_C_b_matches_phase0_validation_reproduction():
    C_a, C_b, T_K, Q_dot = solve_steady_state(
        F=35.0, T_R=THETA, beta=1.0, alpha=1.0, C_A0=5.1, T_in=130.0, C=C
    )
    assert C_b == pytest.approx(1.078, rel=1e-2)


@pytest.mark.parametrize("F,T_R,beta", [
    (5.0, 100.0, 1.0),
    (35.0, 150.0, 1.0),
    (20.0, 125.0, 0.6),
    (5.0, 150.0, 0.2),
    (35.0, 100.0, 0.4),
])
def test_solution_satisfies_the_four_steady_state_balances(F, T_R, beta):
    alpha, C_A0, T_in = 1.0, 5.1, 130.0
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    K1, K2, K3 = rate_constants(T_R, alpha, beta, C)

    rhs_C_a = F * (C_A0 - C_a) - K1 * C_a - K3 * C_a ** 2
    rhs_C_b = -F * C_b + K1 * C_a - K2 * C_b
    T_dif = T_R - T_K
    rhs_T_R = (
        (K1 * C_a * C["H_R_ab"] + K2 * C_b * C["H_R_bc"] + K3 * C_a ** 2 * C["H_R_ad"])
        / (-C["rho"] * C["Cp"])
        + F * (T_in - T_R)
        + ((C["K_w"] * C["A_R"]) * (-T_dif)) / (C["rho"] * C["Cp"] * C["V_R"])
    )
    rhs_T_K = (Q_dot + C["K_w"] * C["A_R"] * T_dif) / (C["m_k"] * C["Cp_k"])

    assert rhs_C_a == pytest.approx(0.0, abs=1e-8)
    assert rhs_C_b == pytest.approx(0.0, abs=1e-8)
    assert rhs_T_R == pytest.approx(0.0, abs=1e-8)
    assert rhs_T_K == pytest.approx(0.0, abs=1e-8)


def test_solve_steady_state_is_vectorized_over_a_grid():
    F = np.linspace(5.0, 35.0, 4)
    T_R = np.linspace(100.0, 150.0, 3)
    F_grid, T_R_grid = np.meshgrid(F, T_R, indexing="ij")
    C_a, C_b, T_K, Q_dot = solve_steady_state(
        F_grid, T_R_grid, beta=1.0, alpha=1.0, C_A0=5.1, T_in=130.0, C=C
    )
    assert C_a.shape == (4, 3)
    assert C_b.shape == (4, 3)
    assert T_K.shape == (4, 3)
    assert Q_dot.shape == (4, 3)
    assert np.all(np.isfinite(C_a))
    assert np.all(np.isfinite(Q_dot))


def test_build_map_saves_only_physical_quantities(tmp_path):
    out_path = tmp_path / "steady_state_map.npz"
    F_grid = np.linspace(5.0, 35.0, 3)
    T_R_grid = np.linspace(100.0, 150.0, 3)
    betas = np.array([1.0, 0.6, 0.2])

    build_map(F_grid, T_R_grid, betas, alpha=1.0, C_A0=5.1, T_in=130.0,
              C=C, out_path=str(out_path))

    data = np.load(out_path)
    expected_keys = {"F", "T_R", "beta", "C_a", "C_b", "T_K", "Q_dot"}
    assert set(data.keys()) == expected_keys

    n_beta, n_F, n_TR = len(betas), len(F_grid), len(T_R_grid)
    for field in ("C_a", "C_b", "T_K", "Q_dot"):
        assert data[field].shape == (n_beta, n_F, n_TR)
        assert np.all(np.isfinite(data[field]))

    forbidden = {"price_B", "price_A0", "price_energy", "T_R_limit", "mask", "profit"}
    assert forbidden.isdisjoint(data.keys())
