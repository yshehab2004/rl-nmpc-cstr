import numpy as np
import pytest

from open_loop_testing.constraint_margins import compute_margins
from open_loop_testing.feasible_region import classify, FEASIBLE, COOLING_FLOOR


def test_margins_match_classify_sign_at_cooling_floor():
    Q_dot = np.array([-9000.0, -8000.0, -8500.0])
    T_R = np.array([120.0, 120.0, 120.0])
    margins = compute_margins(Q_dot, T_R)
    code = classify(Q_dot, T_R)

    assert (margins["Q_dot_margin"][code == COOLING_FLOOR] < 0).all()
    assert (margins["Q_dot_margin"][code == FEASIBLE] >= 0).all()


def test_T_R_margin_is_positive_below_the_limit():
    margins = compute_margins(np.array([-1000.0]), np.array([130.0]), T_R_max=140.0)
    assert margins["T_R_margin"][0] == pytest.approx(10.0)


def test_T_R_margin_is_negative_above_the_limit():
    margins = compute_margins(np.array([-1000.0]), np.array([145.0]), T_R_max=140.0)
    assert margins["T_R_margin"][0] == pytest.approx(-5.0)


def test_margins_on_the_real_cached_map_are_consistent_with_classify():
    d = np.load("open_loop_testing/steady_state_map.npz")
    T_R_mesh = np.broadcast_to(d["T_R"][None, :], d["Q_dot"][0].shape)
    for i in range(len(d["beta"])):
        margins = compute_margins(d["Q_dot"][i], T_R_mesh)
        code = classify(d["Q_dot"][i], T_R_mesh)
        assert np.all(margins["Q_dot_margin"][code == FEASIBLE] >= 0)
        assert np.all(margins["T_R_margin"][code == FEASIBLE] >= 0)
