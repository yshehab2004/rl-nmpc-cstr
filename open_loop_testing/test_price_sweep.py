import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.price_sweep import sweep_price_ratio

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_sweep_returns_one_row_per_ratio():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    ratios = [0.01, 0.05, 0.5]
    rows = sweep_price_ratio(F_grid, T_R_grid, ratios, beta=1.0, price_B=100.0,
                               price_energy=0.02, **KW)
    assert len(rows) == 3
    assert [r["price_A0_over_price_B"] for r in rows] == ratios


def test_higher_raw_material_cost_does_not_increase_profit():
    F_grid = np.linspace(5.0, 35.0, 31)
    T_R_grid = np.linspace(100.0, 150.0, 51)
    rows = sweep_price_ratio(F_grid, T_R_grid, [0.01, 0.5], beta=1.0, price_B=100.0,
                               price_energy=0.02, **KW)
    assert rows[0]["profit_star"] >= rows[1]["profit_star"]
