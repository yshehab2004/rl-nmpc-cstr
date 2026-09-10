import pytest

from models.params import load_params
from open_loop_testing.switch_beta_vs_C_A0 import switch_beta_for_C_A0

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, T_in=130.0, C=C)


def test_switch_beta_at_nominal_C_A0_matches_f_003():
    result = switch_beta_for_C_A0(5.1, **KW)
    assert result == pytest.approx(0.38505, abs=1e-4)


def test_switch_beta_increases_with_C_A0():
    beta_45 = switch_beta_for_C_A0(4.5, **KW)
    beta_57 = switch_beta_for_C_A0(5.7, **KW)
    assert beta_45 < 0.38505 < beta_57
