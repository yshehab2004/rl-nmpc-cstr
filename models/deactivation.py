import math

from models.params import load_params

PARAMS = load_params()
UNCERTAIN = PARAMS["uncertain_params"]

ALPHA_LOCKED = UNCERTAIN["alpha_nominal"]
BETA_NOMINAL = UNCERTAIN["beta_nominal"]

SEVERITY = {
    "mild": 0.50,
    "moderate": 0.35,
    "severe": 0.20,
}


def make_deactivation_fn(beta_final: float, campaign_h: float,
                          profile: str = "exponential",
                          beta_start: float = None):
    beta_start = BETA_NOMINAL if beta_start is None else float(beta_start)
    if not 0.0 < beta_final <= BETA_NOMINAL:
        raise ValueError(f"beta_final must be in (0, {BETA_NOMINAL}], got {beta_final}")
    if not 0.0 < beta_start <= BETA_NOMINAL:
        raise ValueError(f"beta_start must be in (0, {BETA_NOMINAL}], got {beta_start}")
    if beta_final > beta_start:
        raise ValueError(
            f"beta_final {beta_final} exceeds beta_start {beta_start}: a "
            f"catalyst deactivates, it does not recover.")
    if campaign_h <= 0:
        raise ValueError(f"campaign_h must be positive, got {campaign_h}")

    if profile == "exponential":
        if beta_final == beta_start:
            k = 0.0
        else:
            k = math.log(beta_start / beta_final) / campaign_h

        def beta_of(t):
            return max(beta_start * math.exp(-k * max(t, 0.0)), beta_final)

    elif profile == "linear":
        rate = (beta_start - beta_final) / campaign_h

        def beta_of(t):
            return max(beta_start - rate * max(t, 0.0), beta_final)

    else:
        raise ValueError(f"profile must be 'exponential' or 'linear', got {profile!r}")

    def ageing_fn(t_now: float):
        return ALPHA_LOCKED, beta_of(float(t_now))

    ageing_fn.beta_final = beta_final
    ageing_fn.beta_start = beta_start
    ageing_fn.campaign_h = campaign_h
    ageing_fn.profile = profile
    return ageing_fn


def make_severity_fn(level: str, campaign_h: float, profile: str = "exponential"):
    if level not in SEVERITY:
        raise ValueError(f"level must be one of {sorted(SEVERITY)}, got {level!r}")
    return make_deactivation_fn(SEVERITY[level], campaign_h, profile=profile)


if __name__ == "__main__":
    campaign_h = 75.0
    print(f"alpha LOCKED at {ALPHA_LOCKED} in every schedule below.\n")
    for profile in ("exponential", "linear"):
        print(f"--- {profile} profile, {campaign_h} h campaign ---")
        header = "    t(h) | " + " | ".join(f"{lvl:>10}" for lvl in SEVERITY)
        print(header)
        fns = {lvl: make_severity_fn(lvl, campaign_h, profile) for lvl in SEVERITY}
        for t in (0.0, 7.5, 18.75, 37.5, 56.25, 75.0):
            cells = []
            for lvl in SEVERITY:
                a, b = fns[lvl](t)
                cells.append(f"a={a:.2f} b={b:.3f}")
            print(f"   {t:5.1f} | " + " | ".join(f"{c:>10}" for c in cells))
        print()
