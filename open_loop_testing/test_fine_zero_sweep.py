import pytest

from models.params import load_params
from open_loop_testing.fine_zero_sweep import sweep

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"], price_energy=ECON["price_energy"])
F_GRID = [5.0 + 0.25 * i for i in range(121)]
T_R_GRID = [100.0 + 0.25 * i for i in range(201)]


def test_reproduces_F021_T4_at_beta_02():
    result = sweep([0.2], F_GRID, T_R_GRID, **KW, **PRICES)[0]
    assert result["F_star"] == pytest.approx(18.25, abs=0.13)
    assert result["min_rhp_zero"] == pytest.approx(1.083, abs=0.01)
    assert result["bandwidth_cap_h_inv"] == pytest.approx(0.361, abs=0.01)


def test_reproduces_F021_T4_none_at_beta_04():
    result = sweep([0.4], F_GRID, T_R_GRID, **KW, **PRICES)[0]
    assert result["min_rhp_zero"] is None
    assert result["bandwidth_cap_h_inv"] is None


def test_sweep_returns_one_result_per_beta():
    results = sweep([0.3, 0.25, 0.2], F_GRID, T_R_GRID, **KW, **PRICES)
    assert [r["beta"] for r in results] == [0.3, 0.25, 0.2]
