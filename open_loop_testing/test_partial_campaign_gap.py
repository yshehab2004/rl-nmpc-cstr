import numpy as np

from open_loop_testing.partial_campaign_gap import time_at_beta


def test_time_at_beta_matches_analytic_exponential_solution():
    beta_final = 0.20
    campaign_h = 75.0
    k = np.log(1.0 / beta_final) / campaign_h

    def schedule(t):
        beta = max(1.0 * np.exp(-k * max(t, 0.0)), beta_final)
        return (1.0, beta)

    beta_min = 0.30
    t_expected = np.log(1.0 / beta_min) / k
    t = time_at_beta(schedule, beta_min, campaign_h)
    assert abs(t - t_expected) < 1e-4


def test_time_at_beta_returns_campaign_h_when_schedule_never_drops_below():
    def schedule(t):
        return (1.0, 0.9)

    t = time_at_beta(schedule, 0.5, 75.0)
    assert t == 75.0


def test_time_at_beta_returns_campaign_h_for_beta_min_equal_to_beta_final():
    beta_final = 0.20
    campaign_h = 75.0
    k = np.log(1.0 / beta_final) / campaign_h

    def schedule(t):
        beta = max(1.0 * np.exp(-k * max(t, 0.0)), beta_final)
        return (1.0, beta)

    t = time_at_beta(schedule, beta_final, campaign_h)
    assert abs(t - campaign_h) < 1e-3
