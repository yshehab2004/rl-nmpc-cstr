import pytest

from models.params import load_params
from open_loop_testing.switch_beta_vs_fmax import switch_beta_for_fmax

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_switch_beta_at_current_f_max_matches_f_003():
    result = switch_beta_for_fmax(35.0, **KW)
    assert result == pytest.approx(0.38505, abs=1e-4)


def test_switch_beta_increases_with_f_max():
    beta_at_30 = switch_beta_for_fmax(30.0, **KW)
    beta_at_40 = switch_beta_for_fmax(40.0, **KW)
    assert beta_at_40 > beta_at_30
