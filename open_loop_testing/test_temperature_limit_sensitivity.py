import pytest

from models.params import load_params
from open_loop_testing.temperature_limit_sensitivity import switch_beta_for_T_lim, trajectory_for_T_lim

PARAMS = load_params()
C = PARAMS["certain_params"]
KW = dict(alpha=1.0, C_A0=5.1, T_in=130.0, C=C)


def test_switch_beta_reproduces_f_003_at_T_lim_140():
    result = switch_beta_for_T_lim(140.0, **KW)
    assert result == pytest.approx(0.38505, abs=1e-4)


def test_switch_beta_is_monotonically_increasing_as_T_lim_tightens():
    beta_135 = switch_beta_for_T_lim(135.0, **KW)
    beta_140 = switch_beta_for_T_lim(140.0, **KW)
    beta_145 = switch_beta_for_T_lim(145.0, **KW)
    assert beta_135 > beta_140 > beta_145


def test_trajectory_respects_the_given_T_lim():
    rows = trajectory_for_T_lim(145.0, [1.0, 0.6, 0.2], **KW)
    for row in rows:
        assert row["T_R_star"] <= 145.0 + 1e-6
