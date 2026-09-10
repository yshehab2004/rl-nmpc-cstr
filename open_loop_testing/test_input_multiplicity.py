import numpy as np
import pytest

from models.params import load_params
from evaluation.phase0_validation import rate_constants, steady_state_concentrations
from open_loop_testing.input_multiplicity import F_peak, crossover_beta_for_bound

PARAMS = load_params()
C = PARAMS["certain_params"]


@pytest.mark.parametrize("T_R,beta", [
    (134.14, 1.0), (134.14, 0.4), (134.14, 0.2),
    (100.0, 1.0), (120.0, 0.4), (140.0, 0.2),
])
def test_F_peak_matches_brute_force_argmax_of_C_b_vs_F(T_R, beta):
    k1, k2, k3 = rate_constants(T_R, alpha=1.0, beta=beta)
    F = np.linspace(0.5, 2 * F_peak(T_R, 1.0, beta, C) + 5, 400000)
    _, C_b = steady_state_concentrations(F, 5.1, k1, k2, k3)
    F_peak_numeric = F[np.argmax(C_b)]
    assert F_peak_numeric == pytest.approx(F_peak(T_R, 1.0, beta, C), abs=0.05)


def test_F_peak_is_linear_in_beta_at_fixed_T_R():
    p1 = F_peak(134.14, 1.0, 1.0, C)
    p_half = F_peak(134.14, 1.0, 0.5, C)
    assert p_half == pytest.approx(p1 / 2, rel=1e-9)


def test_crossover_beta_reproduces_F_peak_35_at_reference_T_R():
    beta_cross = crossover_beta_for_bound(35.0, 134.14, 1.0, C)
    assert F_peak(134.14, 1.0, beta_cross, C) == pytest.approx(35.0, rel=1e-9)


def test_real_optimum_trajectory_crosses_the_peak_between_beta_0_6_and_0_4():
    from open_loop_testing.optimum_trajectory import find_optimum
    PRICES = dict(price_B=PARAMS["economics"]["price_B"], price_A0=PARAMS["economics"]["price_A0"],
                  price_energy=PARAMS["economics"]["price_energy"])
    KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
    F_grid = np.linspace(5.0, 35.0, 121)
    T_R_grid = np.linspace(100.0, 150.0, 201)

    opt_06 = find_optimum(F_grid, T_R_grid, 0.6, **KW, **PRICES)
    opt_04 = find_optimum(F_grid, T_R_grid, 0.4, **KW, **PRICES)

    assert opt_06["F_star"] < F_peak(opt_06["T_R_star"], 1.0, 0.6, C)
    assert opt_04["F_star"] > F_peak(opt_04["T_R_star"], 1.0, 0.4, C)
