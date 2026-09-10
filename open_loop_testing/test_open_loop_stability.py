import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.open_loop_stability import max_real_eigenvalue

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_max_real_eigenvalue_is_negative_at_the_nominal_point():
    result = max_real_eigenvalue(18.83, 134.14, 1.0, **KW)
    assert result < 0


def test_max_real_eigenvalue_is_vectorizable_over_a_small_grid():
    F_grid = np.linspace(5.0, 35.0, 4)
    T_R_grid = np.linspace(100.0, 150.0, 5)
    results = np.array([[max_real_eigenvalue(F, T_R, 0.6, **KW) for T_R in T_R_grid] for F in F_grid])
    assert results.shape == (4, 5)
    assert np.all(np.isfinite(results))
