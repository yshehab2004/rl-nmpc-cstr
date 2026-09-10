

def time_at_beta(schedule, beta_min, campaign_h, tol=1e-9):
    if schedule(campaign_h)[1] >= beta_min:
        return campaign_h
    lo, hi = 0.0, campaign_h
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if schedule(mid)[1] > beta_min:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
