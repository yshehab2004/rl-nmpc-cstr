import pickle

import numpy as np
import pytest

from rl.env import RandomAgeing, RandomFeedDisturbance

CAMPAIGN_H = 75.0
EVAL_C_A0_STEP = 0.6
EVAL_T_IN_STEP = 3.0
NOMINAL_C_A0 = 5.1
NOMINAL_T_IN = 130.0


def _sample_extremes(sampler, n_episodes=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, CAMPAIGN_H, 400)
    dC, dT = [], []
    for _ in range(n_episodes):
        fn = sampler(rng, CAMPAIGN_H)
        vals = np.array([fn(x) for x in t])
        dC.append(vals[:, 0] - NOMINAL_C_A0)
        dT.append(vals[:, 1] - NOMINAL_T_IN)
    return np.array(dC), np.array(dT)


def test_training_reaches_the_evaluation_C_A0_step_magnitude():
    dC, _ = _sample_extremes(RandomFeedDisturbance())
    assert dC.max() >= EVAL_C_A0_STEP, (
        f"training never reaches the +{EVAL_C_A0_STEP} evaluation step "
        f"(max seen {dC.max():.3f}); T3 would be extrapolating")


def test_training_moves_T_in_at_all_and_reaches_its_evaluation_step():
    _, dT = _sample_extremes(RandomFeedDisturbance())
    assert np.abs(dT).max() > 0.0, "T_in never moves during training"
    assert dT.max() >= EVAL_T_IN_STEP


def test_T_in_drifts_in_every_episode_not_only_when_an_event_fires():
    rng = np.random.default_rng(17)
    t = np.linspace(0.0, CAMPAIGN_H, 300)
    for _ in range(10):
        fn = RandomFeedDisturbance()(rng, CAMPAIGN_H)
        T_in = np.array([fn(x)[1] for x in t])
        assert T_in.std() > 1e-3


def _mean_shift(series):
    n = len(series) // 5
    return float(np.mean(series[-n:]) - np.mean(series[:n]))


def test_at_most_one_large_event_per_episode():
    rng = np.random.default_rng(23)
    t = np.linspace(0.0, CAMPAIGN_H, 500)
    for _ in range(40):
        fn = RandomFeedDisturbance()(rng, CAMPAIGN_H)
        vals = np.array([fn(x) for x in t])
        big_C = abs(_mean_shift(vals[:, 0])) > 0.15
        big_T = abs(_mean_shift(vals[:, 1])) > 0.80
        assert not (big_C and big_T), "C_A0 and T_in events fired together"


def test_every_episode_carries_background_variation():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, CAMPAIGN_H, 300)
    for _ in range(12):
        fn = RandomFeedDisturbance()(rng, CAMPAIGN_H)
        C_A0 = np.array([fn(x)[0] for x in t])
        assert C_A0.std() > 1e-3


def test_some_episodes_have_no_discrete_event():
    rng = np.random.default_rng(5)
    quiet = 0
    for _ in range(40):
        fn = RandomFeedDisturbance()(rng, CAMPAIGN_H)
        t = np.linspace(0.0, CAMPAIGN_H, 200)
        span = np.ptp([fn(x)[0] for x in t])
        if span < 0.35:
            quiet += 1
    assert quiet > 0, "every sampled episode contained a discrete event"


def test_event_timing_is_not_always_the_same_instant():
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, CAMPAIGN_H, 600)
    jump_times = []
    for _ in range(40):
        fn = RandomFeedDisturbance(p_event=1.0)(rng, CAMPAIGN_H)
        C_A0 = np.array([fn(x)[0] for x in t])
        jump_times.append(t[1:][np.argmax(np.abs(np.diff(C_A0)))])
    assert len(set(np.round(jump_times, 1))) > 5


def test_the_exact_evaluation_scenario_is_never_generated():
    rng = np.random.default_rng(29)
    t = np.linspace(0.0, CAMPAIGN_H, 800)
    for _ in range(60):
        fn = RandomFeedDisturbance()(rng, CAMPAIGN_H)
        C_A0 = np.array([fn(x)[0] for x in t])
        d = np.abs(np.diff(C_A0))
        if d.max() < 0.1:
            continue
        onset = t[1:][np.argmax(d)]
        jump = C_A0[1:].max() - C_A0[0]
        exact = abs(onset - 30.0) < 0.5 and abs(jump - 0.6) < 0.02
        assert not exact, "training generated the exact V4/S-scenario step"


def test_ageing_brackets_the_evaluation_endpoints_without_matching_them():
    rng = np.random.default_rng(11)
    finals = [RandomAgeing()(rng, CAMPAIGN_H)(CAMPAIGN_H)[1] for _ in range(300)]
    assert min(finals) < 0.20, (
        f"training never ages past S3's 0.20 endpoint (min {min(finals):.3f}), "
        "so S3 would be extrapolation")
    assert max(finals) < 1.0, "training includes the no-ageing holdout (S1)"
    assert max(finals) > 0.90


def test_ageing_spans_both_sides_of_beta_star():
    rng = np.random.default_rng(13)
    finals = [RandomAgeing()(rng, CAMPAIGN_H)(CAMPAIGN_H)[1] for _ in range(200)]
    assert min(finals) < 0.385 < max(finals)


def test_samplers_are_picklable():
    for obj in (RandomFeedDisturbance(), RandomAgeing()):
        restored = pickle.loads(pickle.dumps(obj))
        rng_a, rng_b = np.random.default_rng(2), np.random.default_rng(2)
        fn_a = obj(rng_a, CAMPAIGN_H)
        fn_b = restored(rng_b, CAMPAIGN_H)
        probe = 0.3 * CAMPAIGN_H
        assert np.allclose(np.ravel(fn_a(probe)), np.ravel(fn_b(probe)))


def test_same_rng_reproduces_the_episode_scenario():
    a = RandomFeedDisturbance()(np.random.default_rng(21), CAMPAIGN_H)
    b = RandomFeedDisturbance()(np.random.default_rng(21), CAMPAIGN_H)
    t = np.linspace(0.0, CAMPAIGN_H, 100)
    assert [a(x) for x in t] == [b(x) for x in t]
