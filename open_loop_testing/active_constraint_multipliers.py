import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.optimum_trajectory import compute_profit


def identify_active_constraints(F, T_R, beta, alpha, C_A0, T_in, C, T_lim=140.0,
                                   F_bounds=(5.0, 35.0), q_lower=-8500.0, tol=1e-3, q_tol=50.0):
    _, _, _, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    active = []
    if abs(F - F_bounds[1]) < tol:
        active.append("F_upper")
    if abs(F - F_bounds[0]) < tol:
        active.append("F_lower")
    if abs(T_R - T_lim) < tol:
        active.append("T_R_upper")
    if abs(Q_dot - q_lower) < q_tol:
        active.append("Q_dot_lower")
    return active


def _F_for_cooling_floor(T_R, beta, alpha, C_A0, T_in, C, q_lower=-8500.0,
                           F_lo=5.0, F_hi=35.0, maxit=100):
    def slack(F):
        _, _, _, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
        return Q_dot - q_lower

    lo, hi = F_lo, F_hi
    slack_lo, slack_hi = slack(lo), slack(hi)
    if slack_lo < 0:
        return F_lo
    if slack_hi > 0:
        return F_hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        s = slack(mid)
        if s > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _T_R_for_cooling_floor(F, beta, alpha, C_A0, T_in, C, q_lower=-8500.0,
                             T_R_lo=100.0, T_R_hi=150.0, maxit=100):
    def slack(T_R):
        _, _, _, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
        return Q_dot - q_lower

    lo, hi = T_R_lo, T_R_hi
    slack_lo, slack_hi = slack(lo), slack(hi)
    if slack_lo >= 0:
        return lo
    if slack_hi < 0:
        return T_R_hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        if slack(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _vertex_point(active, F_max, T_lim, beta, alpha, C_A0, T_in, C, q_lower=-8500.0):
    active = set(active)
    if active == {"F_upper", "T_R_upper"}:
        return F_max, T_lim
    if active == {"F_upper", "Q_dot_lower"}:
        return F_max, _T_R_for_cooling_floor(F_max, beta, alpha, C_A0, T_in, C, q_lower=q_lower)
    if active == {"T_R_upper", "Q_dot_lower"}:
        return _F_for_cooling_floor(T_lim, beta, alpha, C_A0, T_in, C, q_lower=q_lower), T_lim
    raise ValueError(f"no continuous vertex solver for active set {active}")


def _profit_at(F, T_R, beta, alpha, C_A0, T_in, C, price_B, price_A0, price_energy):
    _, C_b, _, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    return compute_profit(F, C_b, C_A0, Q_dot, price_B, price_A0, price_energy)


def _free_T_R_optimum(F, beta, alpha, C_A0, T_in, C, price_B, price_A0, price_energy,
                        T_R_lo=100.0, T_R_hi=150.0, n=20001):
    T_R_coarse = np.linspace(T_R_lo, T_R_hi, n)
    profits = np.array([_profit_at(F, T_R, beta, alpha, C_A0, T_in, C, price_B, price_A0, price_energy)
                          for T_R in T_R_coarse])
    i = np.argmax(profits)
    lo = T_R_coarse[max(i - 1, 0)]
    hi = T_R_coarse[min(i + 1, n - 1)]
    T_R_fine = np.linspace(lo, hi, 2001)
    profits_fine = np.array([_profit_at(F, T_R, beta, alpha, C_A0, T_in, C, price_B, price_A0, price_energy)
                               for T_R in T_R_fine])
    j = np.argmax(profits_fine)
    return T_R_fine[j], profits_fine[j]


def _reduced_second_derivative(F, T_R_star, beta, alpha, C_A0, T_in, C,
                                  price_B, price_A0, price_energy, h=0.05):
    kw = dict(alpha=alpha, C_A0=C_A0, T_in=T_in, C=C)
    prices = dict(price_B=price_B, price_A0=price_A0, price_energy=price_energy)
    p0 = _profit_at(F, T_R_star, beta, **kw, **prices)
    pp = _profit_at(F, T_R_star + h, beta, **kw, **prices)
    pm = _profit_at(F, T_R_star - h, beta, **kw, **prices)
    return (pp - 2 * p0 + pm) / h ** 2


def compute_multipliers(F, T_R, beta, alpha, C_A0, T_in, C, price_B, price_A0, price_energy,
                          T_lim=140.0, F_max=35.0, q_lower=-8500.0, h=0.01, force_constraints=None):
    kw = dict(alpha=alpha, C_A0=C_A0, T_in=T_in, C=C)
    prices = dict(price_B=price_B, price_A0=price_A0, price_energy=price_energy)

    active = force_constraints if force_constraints is not None else \
        identify_active_constraints(F, T_R, beta, T_lim=T_lim, F_bounds=(5.0, F_max), q_lower=q_lower, **kw)

    if active == ["F_upper"]:
        T_R_star, profit_base = _free_T_R_optimum(F_max, beta, **kw, **prices)
        reduced_d2 = _reduced_second_derivative(F_max, T_R_star, beta, **kw, **prices)
        _, profit_relaxed = _free_T_R_optimum(F_max + h, beta, **kw, **prices)
        return {"profit_base": profit_base, "vertex": (F_max, T_R_star),
                "active_constraints": active, "reduced_d2_profit_dT_R2": reduced_d2,
                "multipliers": {"F_upper": (profit_relaxed - profit_base) / h}}

    if len(active) < 2:
        return {"active_constraints": active,
                "multipliers": {name: float("nan") for name in active}}

    base_F, base_T_R = _vertex_point(active, F_max, T_lim, beta, **kw, q_lower=q_lower)
    profit_base = _profit_at(base_F, base_T_R, beta, **kw, **prices)

    multipliers = {}
    for name in active:
        if name == "F_upper":
            F2, T2 = _vertex_point(active, F_max + h, T_lim, beta, **kw, q_lower=q_lower)
        elif name == "T_R_upper":
            F2, T2 = _vertex_point(active, F_max, T_lim + h, beta, **kw, q_lower=q_lower)
        elif name == "Q_dot_lower":
            F2, T2 = _vertex_point(active, F_max, T_lim, beta, **kw, q_lower=q_lower - h)
        else:
            multipliers[name] = float("nan")
            continue
        profit_relaxed = _profit_at(F2, T2, beta, **kw, **prices)
        multipliers[name] = (profit_relaxed - profit_base) / h

    return {"profit_base": profit_base, "vertex": (base_F, base_T_R),
            "active_constraints": active, "multipliers": multipliers}


if __name__ == "__main__":
    import json
    import time

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    econ = PARAMS["economics"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    prices = dict(price_B=econ["price_B"], price_A0=econ["price_A0"], price_energy=econ["price_energy"])

    with open("open_loop_testing/optimum_trajectory.json") as f:
        table = json.load(f)

    print(f"{'beta':>6} {'F*':>7} {'T_R*':>8} {'active constraints':>30} {'multipliers':>55}")
    results = []
    for row in table:
        result = compute_multipliers(row["F_star"], row["T_R_star"], row["beta"], **kw, **prices)
        results.append({"beta": row["beta"], **result})
        mult_str = ", ".join(f"{k}={v:.3f}" for k, v in result["multipliers"].items())
        print(f"{row['beta']:6.3f} {row['F_star']:7.2f} {row['T_R_star']:8.2f} "
              f"{str(result['active_constraints']):>30} {mult_str:>55}")

    out_path = "open_loop_testing/active_constraint_multipliers.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None, params={"h": 0.01, "T_lim": 140.0}, script=__file__, start_time=start_time,
        extra={"results": results},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")
