import numpy as np
import pytest

from rl.env import ActionMapper, SupervisoryEnv
from rl.policy_supervisor import RLSupervisor

CAMPAIGN_H = 75.0


def _save_untrained(tmp_path_factory, observe_cooling_margin, tag):
    from stable_baselines3 import SAC

    env = SupervisoryEnv(campaign_h=0.5, decision_interval_h=0.25, n_horizon=15,
                          observe_cooling_margin=observe_cooling_margin)
    model = SAC("MlpPolicy", env, seed=0, learning_starts=1, buffer_size=100)
    path = tmp_path_factory.mktemp(tag) / "model.zip"
    model.save(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_model_no_margin(tmp_path_factory):
    return _save_untrained(tmp_path_factory, False, "policy_nm")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from stable_baselines3 import SAC

    env = SupervisoryEnv(campaign_h=0.5, decision_interval_h=0.25, n_horizon=15)
    model = SAC("MlpPolicy", env, seed=0, learning_starts=1, buffer_size=100)
    path = tmp_path_factory.mktemp("policy") / "model.zip"
    model.save(path)
    return str(path)


def _context(Q_dot=-4000.0, F=18.0, beta=0.8):
    return {"t": 12.0, "beta_hat": beta, "C_A0": 5.2, "T_in": 130.5,
            "x_hat": np.array([2.0, 1.1, 134.0, 128.0]),
            "u_last": np.array([F, Q_dot])}


def test_returns_the_supervisory_triple(tiny_model):
    sup = RLSupervisor(tiny_model, campaign_h=CAMPAIGN_H)
    ctx = _context()
    target = sup.get_target(ctx["t"], beta_hat=ctx["beta_hat"], C_A0=ctx["C_A0"],
                             T_in=ctx["T_in"], x_hat=ctx["x_hat"],
                             u_last=ctx["u_last"])
    assert len(target) == 3
    m = ActionMapper()
    F_ref, T_R_ref, margin = target
    assert m.F_lo <= F_ref <= m.F_hi
    assert m.T_R_lo <= T_R_ref <= m.T_R_hi
    assert m.margin_lo <= margin <= m.margin_hi


def test_observation_is_built_by_the_environments_own_code(tiny_model):
    sup = RLSupervisor(tiny_model, campaign_h=CAMPAIGN_H)
    env = SupervisoryEnv(campaign_h=CAMPAIGN_H, decision_interval_h=0.25,
                          observe_cooling_margin=True)
    ctx = _context()
    assert np.array_equal(sup.observation_for(ctx), env.build_observation(ctx))


def test_loading_a_policy_under_the_WRONG_ablation_arm_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="observe_cooling_margin"):
        RLSupervisor(tiny_model, campaign_h=CAMPAIGN_H,
                      observe_cooling_margin=False)


def test_ablation_arms_have_different_observation_widths(
        tiny_model, tiny_model_no_margin):
    with_m = RLSupervisor(tiny_model, campaign_h=CAMPAIGN_H,
                           observe_cooling_margin=True)
    without = RLSupervisor(tiny_model_no_margin, campaign_h=CAMPAIGN_H,
                            observe_cooling_margin=False)
    ctx = _context()
    assert len(with_m.observation_for(ctx)) == len(without.observation_for(ctx)) + 1


def test_actions_are_deterministic_at_evaluation(tiny_model):
    sup = RLSupervisor(tiny_model, campaign_h=CAMPAIGN_H)
    ctx = _context()
    kw = dict(beta_hat=ctx["beta_hat"], C_A0=ctx["C_A0"], T_in=ctx["T_in"],
              x_hat=ctx["x_hat"], u_last=ctx["u_last"])
    first = sup.get_target(ctx["t"], **kw)
    for _ in range(4):
        assert sup.get_target(ctx["t"], **kw) == pytest.approx(first)


def test_campaign_h_mismatch_is_rejected_rather_than_silently_rescaled(tiny_model):
    sup = RLSupervisor(tiny_model, campaign_h=CAMPAIGN_H)
    with pytest.raises(ValueError, match="campaign"):
        sup.get_target(120.0, beta_hat=0.5, C_A0=5.1, T_in=130.0,
                        x_hat=np.array([2.0, 1.1, 134.0, 128.0]),
                        u_last=np.array([18.0, -4000.0]))

    allowed = RLSupervisor(tiny_model, campaign_h=CAMPAIGN_H,
                            allow_overrun=True)
    target = allowed.get_target(120.0, beta_hat=0.5, C_A0=5.1, T_in=130.0,
                                 x_hat=np.array([2.0, 1.1, 134.0, 128.0]),
                                 u_last=np.array([18.0, -4000.0]))
    assert len(target) == 3


def test_drops_into_run_campaign_like_any_other_tier(tiny_model):
    from supervisors.campaign import run_campaign

    sup = RLSupervisor(tiny_model, campaign_h=0.5)
    run = run_campaign(sup, campaign_h=0.5, decision_interval_h=0.25,
                        seed=1, n_horizon=15)
    assert np.all(np.isfinite(run["x"]))
    assert run["metrics"]["total_profit"] == pytest.approx(
        run["metrics"]["total_profit"])
