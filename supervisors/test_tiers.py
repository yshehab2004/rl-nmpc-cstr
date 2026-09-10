import numpy as np
import pytest

from models.deactivation import make_deactivation_fn
from models.params import load_params
from supervisors.campaign import run_campaign
from supervisors.tiers import (RTOSupervisor, T0a_naive, T0b_prudent,
                                margin_for_beta)

PARAMS = load_params()
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]


def test_T0a_is_the_known_infeasible_fresh_optimum():
    sup = T0a_naive()
    assert sup.get_target(0.0) == (35.0, 133.0, 0.0)
    assert sup.get_target(50.0, beta_hat=0.2, C_A0=5.7) == (35.0, 133.0, 0.0)


def test_T0b_is_the_always_feasible_point_with_the_derived_margin():
    sup = T0b_prudent()
    assert sup.get_target(0.0) == (18.25, 140.0, 2.5)
    assert sup.get_target(70.0, beta_hat=0.2) == (18.25, 140.0, 2.5)


def test_margin_schedule_matches_the_derived_thresholds():
    assert margin_for_beta(1.0) == 0.0
    assert margin_for_beta(0.6) == 0.0
    assert margin_for_beta(0.5) == 2.0
    assert margin_for_beta(0.4) == 2.0
    assert margin_for_beta(0.35) == 2.25
    assert margin_for_beta(0.2) == 2.5
    assert margin_for_beta(0.1) == 2.5


def test_margin_schedule_is_monotone_as_ageing_worsens():
    betas = np.linspace(1.0, 0.05, 60)
    margins = [margin_for_beta(b) for b in betas]
    assert all(b >= a for a, b in zip(margins, margins[1:]))


def test_rto_respects_its_cadence():
    sup = RTOSupervisor(cadence_h=2.0)
    sup.get_target(0.0, beta_hat=1.0)
    assert sup.n_solves == 1
    sup.get_target(0.25, beta_hat=1.0)
    sup.get_target(1.75, beta_hat=1.0)
    assert sup.n_solves == 1
    sup.get_target(2.0, beta_hat=1.0)
    assert sup.n_solves == 2


def test_rto_uses_the_pessimistic_end_of_the_estimate():
    sup = RTOSupervisor(cadence_h=2.0)
    sup.get_target(0.0, beta_hat=0.50)
    assert sup.history[0]["beta_eval"] == pytest.approx(0.46)


def test_rto_moves_its_target_as_the_estimate_ages():
    sup = RTOSupervisor(cadence_h=2.0)
    fresh = sup.get_target(0.0, beta_hat=1.0)
    aged = sup.get_target(2.0, beta_hat=0.25)
    assert aged != fresh
    assert aged[0] < fresh[0]
    assert aged[2] > fresh[2]


@pytest.fixture(scope="module")
def short_ageing():
    return make_deactivation_fn(beta_final=0.6, campaign_h=4.0)


def test_campaign_runs_and_reports_metrics(short_ageing):
    run = run_campaign(T0b_prudent(), campaign_h=0.5, ageing_fn=short_ageing,
                        seed=1, n_horizon=20)
    assert np.all(np.isfinite(run["x"]))
    assert np.all(np.isfinite(run["u"]))
    for key in ("total_profit", "violation_max", "beta_hat_max_error"):
        assert key in run["metrics"]


def test_every_tier_honours_the_hard_cooling_bound(short_ageing):
    for supervisor in (T0a_naive(), T0b_prudent(), RTOSupervisor(cadence_h=0.25)):
        run = run_campaign(supervisor, campaign_h=0.5, ageing_fn=short_ageing,
                            seed=2, n_horizon=20)
        Q_dot = run["u"][:, 1]
        assert Q_dot.min() >= -8500.0 - 1e-3
        assert Q_dot.max() <= 0.0 + 1e-3


def test_prudent_tier_violates_less_than_the_naive_one(short_ageing):
    naive = run_campaign(T0a_naive(), campaign_h=0.5, ageing_fn=short_ageing,
                          seed=3, n_horizon=20)
    prudent = run_campaign(T0b_prudent(), campaign_h=0.5, ageing_fn=short_ageing,
                            seed=3, n_horizon=20)
    assert prudent["metrics"]["violation_integral_degC_h"] <= \
        naive["metrics"]["violation_integral_degC_h"]


def test_rto_retargets_over_a_campaign_while_fixed_tiers_do_not(short_ageing):
    rto = RTOSupervisor(cadence_h=0.25)
    run = run_campaign(rto, campaign_h=1.0, ageing_fn=short_ageing,
                        seed=4, n_horizon=20)
    assert rto.n_solves > 1

    assert len(set(np.round(run["T_R_ref"], 6))) > 1
    assert len(set(np.round(run["F_ref"], 6))) == 1

    fixed = T0b_prudent()
    run_fixed = run_campaign(fixed, campaign_h=1.0, ageing_fn=short_ageing,
                              seed=4, n_horizon=20)
    assert len(set(np.round(run_fixed["T_R_ref"], 6))) == 1
    assert len(set(np.round(run_fixed["F_ref"], 6))) == 1
