import pickle

import numpy as np
import pytest

from models.disturbances import (build_feed_disturbance, ou_noise_disturbance,
                                  ramp_disturbance, step_disturbance)

BASE = 5.1
T_MAX = 75.0


def test_noise_is_deterministic_in_time():
    gen = ou_noise_disturbance(BASE, std=0.05, tau_h=0.5, seed=1, t_max=T_MAX)
    probe = [0.0, 0.137, 3.5, 3.5, 0.137, 40.0, 74.9, 3.5]
    first = [gen(t) for t in probe]
    second = [gen(t) for t in reversed(probe)]
    assert first == list(reversed(second))


def test_different_seeds_give_different_realisations():
    gen1 = ou_noise_disturbance(BASE, std=0.05, tau_h=0.5, seed=1, t_max=T_MAX)
    gen2 = ou_noise_disturbance(BASE, std=0.05, tau_h=0.5, seed=2, t_max=T_MAX)
    t = np.linspace(0.0, T_MAX, 500)
    assert not np.allclose([gen1(x) for x in t], [gen2(x) for x in t])


def test_same_seed_reproduces_the_realisation():
    a = ou_noise_disturbance(BASE, std=0.05, tau_h=0.5, seed=7, t_max=T_MAX)
    b = ou_noise_disturbance(BASE, std=0.05, tau_h=0.5, seed=7, t_max=T_MAX)
    t = np.linspace(0.0, T_MAX, 200)
    assert [a(x) for x in t] == [b(x) for x in t]


def test_std_matches_the_requested_value():
    std = 0.08
    gen = ou_noise_disturbance(BASE, std=std, tau_h=0.5, seed=3, t_max=T_MAX)
    samples = np.array([gen(t) for t in np.linspace(0.0, T_MAX, 20000)])
    assert samples.mean() == pytest.approx(BASE, abs=0.05)
    assert samples.std() == pytest.approx(std, rel=0.15)


def test_noise_is_correlated_not_white():
    tau = 0.5
    gen = ou_noise_disturbance(BASE, std=0.05, tau_h=tau, seed=4, t_max=T_MAX)
    dt = 0.01
    x = np.array([gen(t) for t in np.arange(0.0, T_MAX, dt)]) - BASE
    lag = int(round(tau / dt))
    rho = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    assert rho == pytest.approx(np.exp(-1.0), abs=0.12)


def test_zero_std_reproduces_the_constant_feed():
    gen = ou_noise_disturbance(BASE, std=0.0, tau_h=0.5, seed=5, t_max=T_MAX)
    assert all(gen(t) == BASE for t in np.linspace(0.0, T_MAX, 100))


def test_noise_superposes_on_a_step():
    step = step_disturbance(BASE, 0.6, t_start=30.0)
    gen = ou_noise_disturbance(step, std=0.05, tau_h=0.5, seed=6, t_max=T_MAX)
    before = np.array([gen(t) for t in np.linspace(1.0, 29.0, 2000)])
    after = np.array([gen(t) for t in np.linspace(31.0, 74.0, 2000)])
    assert after.mean() - before.mean() == pytest.approx(0.6, abs=0.06)
    assert before.std() > 0.01 and after.std() > 0.01


def test_the_whole_composed_stack_is_picklable():
    fn = build_feed_disturbance(
        C_A0_gen=ou_noise_disturbance(step_disturbance(BASE, 0.6, 30.0),
                                       std=0.05, tau_h=0.5, seed=8, t_max=T_MAX),
        T_in_gen=ramp_disturbance(130.0, 3.0, 10.0, 20.0))
    restored = pickle.loads(pickle.dumps(fn))
    for t in (0.0, 5.0, 15.0, 30.0, 31.0, 74.0):
        assert restored(t) == fn(t)


def test_existing_generators_are_picklable_too():
    for gen in (step_disturbance(BASE, 0.6, 30.0),
                ramp_disturbance(BASE, 0.6, 8.0, 13.0)):
        restored = pickle.loads(pickle.dumps(gen))
        assert [restored(t) for t in (0.0, 9.0, 12.0, 40.0)] == \
               [gen(t) for t in (0.0, 9.0, 12.0, 40.0)]
