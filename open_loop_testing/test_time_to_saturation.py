import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.time_to_saturation import critical_beta

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_critical_beta_is_bracketed_correctly_at_F35_T134():
    result = critical_beta(35.0, 139.0, **KW)
    assert 0.4 < result < 0.6


def test_never_saturates_returns_beta_lo_sentinel():
    result = critical_beta(5.0, 145.0, **KW, beta_lo=0.05, beta_hi=1.0)
    assert result == pytest.approx(0.05)


def test_already_saturated_at_fresh_catalyst_returns_beta_hi_sentinel():
    result = critical_beta(35.0, 100.0, **KW, beta_lo=0.05, beta_hi=1.0)
    assert result == pytest.approx(1.0)


def test_critical_beta_is_vectorized_and_finite_over_the_grid():
    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid = d["F"][::10], d["T_R"][::10]
    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")
    result = critical_beta(F_mesh, T_R_mesh, **KW)
    assert result.shape == F_mesh.shape
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0.05 - 1e-9) & (result <= 1.0 + 1e-9))
