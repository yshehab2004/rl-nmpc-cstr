import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.heat_decomposition import decompose_Q_dot

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


@pytest.mark.parametrize("F,T_R,beta", [
    (20.0, 134.0, 1.0), (20.0, 134.0, 0.6), (20.0, 134.0, 0.2),
    (35.0, 139.0, 1.0), (35.0, 139.0, 0.6), (35.0, 139.0, 0.2),
    (5.0, 100.0, 1.0), (35.0, 150.0, 0.2),
])
def test_feed_plus_reaction_terms_reconstruct_Q_dot_exactly(F, T_R, beta):
    _, _, _, Q_dot = solve_steady_state(F, T_R, beta, **KW)
    terms = decompose_Q_dot(F, T_R, beta, **KW)
    assert terms["Q_feed"] + terms["Q_rxn"] == pytest.approx(Q_dot, rel=1e-9)
    assert terms["Q_rxn"] == pytest.approx(terms["Q_AB"] + terms["Q_BC"] + terms["Q_AD"], rel=1e-9)


def test_feed_term_sign_matches_direction_of_T_R_minus_T_in():
    terms_cold = decompose_Q_dot(20.0, 100.0, 1.0, **KW)
    assert terms_cold["Q_feed"] < 0

    terms_hot = decompose_Q_dot(20.0, 145.0, 1.0, **KW)
    assert terms_hot["Q_feed"] > 0


def test_AD_pathway_is_the_largest_single_reaction_term_at_low_beta():
    terms = decompose_Q_dot(35.0, 139.0, 0.2, **KW)
    assert abs(terms["Q_AD"]) > abs(terms["Q_AB"])
    assert abs(terms["Q_AD"]) > abs(terms["Q_BC"])


def test_reconstructs_Q_dot_from_the_cached_map_array_not_a_fresh_solve():
    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid, betas, Q_dot_map = d["F"], d["T_R"], d["beta"], d["Q_dot"]

    rng = np.random.default_rng(0)
    for _ in range(20):
        i_beta = rng.integers(0, len(betas))
        i_F = rng.integers(0, len(F_grid))
        i_T = rng.integers(0, len(T_R_grid))
        F, T_R, beta = float(F_grid[i_F]), float(T_R_grid[i_T]), float(betas[i_beta])

        terms = decompose_Q_dot(F, T_R, beta, **KW)
        expected = Q_dot_map[i_beta, i_F, i_T]
        assert terms["Q_feed"] + terms["Q_rxn"] == pytest.approx(float(expected), rel=1e-9)


def test_decompose_Q_dot_is_vectorized_over_beta():
    betas = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    terms = decompose_Q_dot(35.0, 139.0, betas, **KW)
    assert terms["Q_feed"].shape == (5,)
    assert terms["Q_rxn"].shape == (5,)
    assert np.allclose(terms["Q_feed"], terms["Q_feed"][0])
