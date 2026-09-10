import numpy as np
import pytest

from models.params import load_params
from open_loop_testing.hessian_conditioning import profit_hessian, condition_number

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"], price_energy=ECON["price_energy"])


def test_hessian_is_symmetric():
    H = profit_hessian(20.0, 130.0, 1.0, **KW, **PRICES)
    assert H[0, 1] == pytest.approx(H[1, 0], rel=1e-6)


def test_condition_number_is_at_least_one():
    H = profit_hessian(20.0, 130.0, 1.0, **KW, **PRICES)
    kappa = condition_number(H)
    assert kappa >= 1.0


def test_condition_number_of_a_known_ridge_matrix():
    H = np.array([[1.0, 0.0], [0.0, 1e-6]])
    assert condition_number(H) == pytest.approx(1e6, rel=1e-3)
