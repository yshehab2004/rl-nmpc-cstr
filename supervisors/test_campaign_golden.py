import numpy as np
import pytest

from models.deactivation import make_deactivation_fn
from models.disturbances import build_feed_disturbance, step_disturbance
from supervisors.campaign import run_campaign
from supervisors.tiers import RTOSupervisor, T0b_prudent

TOL = 1e-6

GOLDEN_A = {
    "x_final": [1.0544503984, 0.8133803709, 137.5000000382, 134.4295777006],
    "u_final": [18.2500000358, -2661.6877159998],
    "total_profit": 470.5996189779,
    "beta_hat_final": 0.999896010146,
}

GOLDEN_B = {
    "x_final": [2.0738110229, 1.1579171372, 137.8783306003, 128.0044193615],
    "u_final": [34.9811855022, -8500.0000063238],
    "total_profit": 1358.1730511923,
    "F_s_final": 35.0,
    "T_R_s_final": 136.5,
}


def test_golden_A_fixed_tier_no_ageing():
    run = run_campaign(T0b_prudent(), campaign_h=0.5, seed=1, n_horizon=15)
    assert run["x"][-1] == pytest.approx(GOLDEN_A["x_final"], abs=TOL)
    assert run["u"][-1] == pytest.approx(GOLDEN_A["u_final"], abs=TOL)
    assert run["metrics"]["total_profit"] == pytest.approx(
        GOLDEN_A["total_profit"], abs=TOL)
    assert float(run["beta_hat"][-1]) == pytest.approx(
        GOLDEN_A["beta_hat_final"], abs=1e-9)


def test_golden_B_rto_under_ageing_and_feed_step():
    run = run_campaign(
        RTOSupervisor(cadence_h=0.25), campaign_h=0.5, seed=3, n_horizon=15,
        ageing_fn=make_deactivation_fn(beta_final=0.7, campaign_h=0.5),
        disturbance_fn=build_feed_disturbance(
            C_A0_gen=step_disturbance(5.1, 0.6, 0.2)))
    assert run["x"][-1] == pytest.approx(GOLDEN_B["x_final"], abs=TOL)
    assert run["u"][-1] == pytest.approx(GOLDEN_B["u_final"], abs=TOL)
    assert run["metrics"]["total_profit"] == pytest.approx(
        GOLDEN_B["total_profit"], abs=TOL)
    assert float(run["F_s"][-1]) == pytest.approx(GOLDEN_B["F_s_final"], abs=TOL)
    assert float(run["T_R_s"][-1]) == pytest.approx(GOLDEN_B["T_R_s_final"], abs=TOL)


def test_campaign_is_reproducible_from_its_seed():
    kw = dict(campaign_h=0.25, seed=11, n_horizon=15)
    a = run_campaign(T0b_prudent(), **kw)
    b = run_campaign(T0b_prudent(), **kw)
    assert np.array_equal(a["x"], b["x"])
    assert np.array_equal(a["u"], b["u"])
