import numpy as np
import pytest

from models.deactivation import make_deactivation_fn
from supervisors.campaign import run_campaign
from supervisors.tiers import RTOSupervisor, T0a_naive, T0b_prudent

X_PROBE_A = np.array([2.14, 1.09, 134.14, 128.5])
X_PROBE_B = np.array([1.80, 1.20, 139.00, 133.0])
U_PROBE = np.array([18.83, -4495.0])


class SpySupervisor:

    def __init__(self):
        self.name = "spy"
        self.n_solves = 0
        self.calls = []

    def get_target(self, t_now, beta_hat=None, C_A0=None, T_in=None,
                    x_hat=None, u_last=None):
        self.calls.append({"t": t_now, "beta_hat": beta_hat, "C_A0": C_A0,
                            "T_in": T_in,
                            "x_hat": None if x_hat is None else np.asarray(x_hat).copy(),
                            "u_last": None if u_last is None else np.asarray(u_last).copy()})
        self.n_solves += 1
        return (18.25, 140.0, 2.5)


@pytest.mark.parametrize("factory", [T0a_naive, T0b_prudent,
                                      lambda: RTOSupervisor(cadence_h=2.0)])
def test_every_tier_accepts_the_full_context(factory):
    sup = factory()
    target = sup.get_target(0.0, beta_hat=0.8, C_A0=5.1, T_in=130.0,
                             x_hat=X_PROBE_A, u_last=U_PROBE)
    assert len(target) == 3


@pytest.mark.parametrize("factory", [T0a_naive, T0b_prudent])
def test_fixed_tiers_ignore_plant_state(factory):
    sup = factory()
    a = sup.get_target(0.0, beta_hat=1.0, C_A0=5.1, T_in=130.0, x_hat=X_PROBE_A)
    b = sup.get_target(5.0, beta_hat=0.3, C_A0=5.7, T_in=133.0, x_hat=X_PROBE_B,
                        u_last=U_PROBE)
    assert a == b


def test_rto_target_depends_on_belief_not_on_plant_state():
    sup_a, sup_b = RTOSupervisor(cadence_h=2.0), RTOSupervisor(cadence_h=2.0)
    kw = dict(beta_hat=0.5, C_A0=5.1, T_in=130.0)
    assert sup_a.get_target(0.0, x_hat=X_PROBE_A, **kw) == \
           sup_b.get_target(0.0, x_hat=X_PROBE_B, u_last=U_PROBE, **kw)


def test_campaign_offers_plant_state_to_the_supervisor():
    spy = SpySupervisor()
    run_campaign(spy, campaign_h=0.75, decision_interval_h=0.25,
                  seed=1, n_horizon=15,
                  ageing_fn=make_deactivation_fn(beta_final=0.9, campaign_h=0.75))
    assert len(spy.calls) >= 3

    for call in spy.calls[1:]:
        assert call["x_hat"] is not None, "plant state was never wired through"
        assert call["x_hat"].shape == (4,)
        assert np.all(np.isfinite(call["x_hat"]))
        assert call["u_last"] is not None, "last input was never wired through"
        assert call["u_last"].shape == (2,)
        assert np.all(np.isfinite(call["u_last"]))


def test_u_last_is_none_only_on_the_first_decision():
    spy = SpySupervisor()
    run_campaign(spy, campaign_h=0.75, decision_interval_h=0.25,
                  seed=1, n_horizon=15)
    assert spy.calls[0]["u_last"] is None
    assert all(c["u_last"] is not None for c in spy.calls[1:])


def test_state_offered_is_the_estimate_not_the_truth():
    spy = SpySupervisor()
    run = run_campaign(spy, campaign_h=0.75, decision_interval_h=0.25,
                        seed=1, noise=True, n_horizon=15)
    t_last = spy.calls[-1]["t"]
    k = int(round(t_last / (run["t"][1] - run["t"][0])))
    x_true = run["x"][k]
    assert not np.allclose(spy.calls[-1]["x_hat"], x_true, atol=1e-9)
