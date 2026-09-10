import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.settling_time import time_to_settle, settling_time_for_step

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_time_to_settle_on_a_synthetic_exponential_decay():
    t = np.linspace(0, 20, 200000)
    x = 10 * np.exp(-t) + 1.0
    result = time_to_settle(t, x, x_final=1.0, tol=0.02, initial_value=11.0)
    assert result == pytest.approx(np.log(50), abs=0.01)


def test_time_to_settle_returns_zero_if_already_within_band():
    t = np.linspace(0, 5, 1000)
    x = np.full_like(t, 1.0005)
    result = time_to_settle(t, x, x_final=1.0, tol=0.02, initial_value=1.5)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_settling_time_for_step_at_nominal_point_is_positive_and_bounded():
    result = settling_time_for_step(18.83, 134.14, 1.0, du=np.array([1.0, 0.0]),
                                      t_max=5.0, **KW)
    assert 0.0 < result["C_b"] < 5.0
    assert 0.0 < result["T_R"] < 5.0
