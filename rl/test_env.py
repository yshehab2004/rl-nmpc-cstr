import pickle

import numpy as np
import pytest

from rl.env import (ActionMapper, F_LOWER, F_UPPER, SupervisoryEnv, make_env,
                     reward_from, ConstantAgeing)

SHORT = dict(campaign_h=0.5, decision_interval_h=0.25, n_horizon=15, seed=1)


def _context(Q_dot=-4000.0, F=18.0):
    return {"t": 0.25, "beta_hat": 0.8, "C_A0": 5.1, "T_in": 130.0,
            "x_hat": np.array([2.0, 1.1, 134.0, 128.0]),
            "u_last": np.array([F, Q_dot])}


def test_observation_matches_the_declared_space():
    env = SupervisoryEnv(**SHORT)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs), (
        f"observation {obs} outside declared space "
        f"{env.observation_space.low}..{env.observation_space.high}")
    assert obs.dtype == np.float32


def test_cooling_margin_flag_changes_the_observation_width():
    with_margin = SupervisoryEnv(observe_cooling_margin=True, **SHORT)
    without = SupervisoryEnv(observe_cooling_margin=False, **SHORT)
    assert (with_margin.observation_space.shape[0]
            == without.observation_space.shape[0] + 1)


def test_without_the_flag_cooling_information_is_absent():
    env = SupervisoryEnv(observe_cooling_margin=False, **SHORT)
    a = env.build_observation(_context(Q_dot=-500.0))
    b = env.build_observation(_context(Q_dot=-8400.0))
    assert np.array_equal(a, b), (
        "Q_dot leaked into the base observation; the cooling-margin "
        "ablation would measure nothing")


def test_with_the_flag_cooling_information_is_present():
    env = SupervisoryEnv(observe_cooling_margin=True, **SHORT)
    a = env.build_observation(_context(Q_dot=-500.0))
    b = env.build_observation(_context(Q_dot=-8400.0))
    assert not np.array_equal(a, b)


def test_observed_F_is_the_APPLIED_input_not_the_commanded_one():
    env = SupervisoryEnv(**SHORT)
    env.reset(seed=1)
    obs, _, _, _, info = env.step(np.array([1.0, 0.0, -1.0], dtype=np.float32))

    F_applied = float(env.engine.hist["u"][-1][0])
    observed = float(obs[7]) * (F_UPPER - F_LOWER) + F_LOWER

    assert observed == pytest.approx(F_applied, abs=1e-6), (
        "observation slot 7 does not carry the applied F")
    assert abs(F_applied - info["F_ref"]) > 1e-3, (
        "test is vacuous: commanded and applied F did not diverge here")


def test_first_observation_reports_no_cooling_used_rather_than_zero_margin():
    env = SupervisoryEnv(observe_cooling_margin=True, **SHORT)
    ctx = _context()
    ctx["u_last"] = None
    obs = env.build_observation(ctx)
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs.astype(np.float32))


def test_action_bounds_map_to_the_supervisory_triple():
    m = ActionMapper()
    lo = m.to_target(np.array([-1.0, -1.0, -1.0]))
    hi = m.to_target(np.array([1.0, 1.0, 1.0]))
    assert lo == pytest.approx((m.F_lo, m.T_R_lo, m.margin_lo))
    assert hi == pytest.approx((m.F_hi, m.T_R_hi, m.margin_hi))


def test_action_is_clipped_not_wrapped():
    m = ActionMapper()
    assert m.to_target(np.array([5.0, 5.0, 5.0])) == pytest.approx(
        m.to_target(np.array([1.0, 1.0, 1.0])))


def test_reward_is_profit_minus_priced_violation():
    r_clean = reward_from(profit=100.0, violation_integral=0.0, weight=250.0,
                           interval_h=0.25)
    r_dirty = reward_from(profit=100.0, violation_integral=0.1, weight=250.0,
                           interval_h=0.25)
    assert r_dirty < r_clean
    assert (r_clean - r_dirty) == pytest.approx(
        reward_from(profit=25.0, violation_integral=0.0, weight=250.0,
                     interval_h=0.25), rel=1e-9)


def test_violating_is_never_profitable_at_the_configured_weight():
    shadow_price = 209.0
    gain = shadow_price * 0.5
    kw = dict(interval_h=0.25, weight=250.0)
    assert reward_from(profit=gain, violation_integral=0.5, **kw) < \
           reward_from(profit=0.0, violation_integral=0.0, **kw)


def test_episode_truncates_at_the_campaign_length():
    env = SupervisoryEnv(**SHORT)
    env.reset(seed=1)
    steps, truncated = 0, False
    while not truncated and steps < 10:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert not terminated, "no divergence expected in a benign 0.5 h episode"
    assert truncated
    assert steps == int(round(SHORT["campaign_h"] / SHORT["decision_interval_h"]))


def test_same_seed_reproduces_the_episode():
    a, b = SupervisoryEnv(**SHORT), SupervisoryEnv(**SHORT)
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)
    act = np.array([0.2, 0.1, -0.5], dtype=np.float32)
    ra = a.step(act)
    rb = b.step(act)
    assert np.array_equal(ra[0], rb[0])
    assert ra[1] == rb[1]


def test_info_carries_the_two_reporting_axes_separately():
    env = SupervisoryEnv(**SHORT)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert "profit" in info and "violation_integral" in info
    assert "F_s" in info and "T_R_s" in info


def test_env_factory_is_picklable():
    factory = make_env(campaign_h=0.5, decision_interval_h=0.25, n_horizon=15,
                        seed=3, observe_cooling_margin=True,
                        ageing=ConstantAgeing(0.8))
    restored = pickle.loads(pickle.dumps(factory))
    env = restored()
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)


def test_the_env_itself_is_not_expected_to_pickle():
    env = SupervisoryEnv(**SHORT)
    env.reset(seed=1)
    env.step(np.zeros(3, dtype=np.float32))
    with pytest.raises(Exception):
        pickle.dumps(env)


def test_passes_the_gymnasium_api_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(SupervisoryEnv(**SHORT), skip_render_check=True)
