import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.optimum_trajectory import compute_profit, find_optimum
from open_loop_testing.feasible_region import classify, FEASIBLE


def _time_samples(campaign_h, n=76):
    return np.linspace(0.0, campaign_h, n)


def campaign_total_profit_fixed(F, T_R, schedule, campaign_h, alpha, C_A0, T_in, C,
                                   price_B, price_A0, price_energy, n=76):
    t = _time_samples(campaign_h, n)
    betas = np.array([schedule(ti)[1] for ti in t])
    _, C_b, _, Q_dot = solve_steady_state(F, T_R, betas, alpha, C_A0, T_in, C)
    code = classify(Q_dot, np.full_like(betas, T_R))
    if np.any(code != FEASIBLE):
        return float("nan")
    profit = compute_profit(F, C_b, C_A0, Q_dot, price_B, price_A0, price_energy)
    return float(np.trapezoid(profit, t))


def campaign_total_profit_adaptive(F_grid, T_R_grid, schedule, campaign_h, alpha, C_A0, T_in, C,
                                      price_B, price_A0, price_energy, n=76):
    t = _time_samples(campaign_h, n)
    profits = np.empty(n)
    for i, ti in enumerate(t):
        beta = schedule(ti)[1]
        opt = find_optimum(F_grid, T_R_grid, beta, alpha, C_A0, T_in, C, price_B, price_A0, price_energy)
        profits[i] = opt["profit_star"]
    return float(np.trapezoid(profits, t))


def find_best_always_feasible_point(F_grid, T_R_grid, schedule, campaign_h, alpha, C_A0, T_in, C,
                                       price_B, price_A0, price_energy, n=76):
    F_grid = np.asarray(F_grid, dtype=float)
    T_R_grid = np.asarray(T_R_grid, dtype=float)
    _, beta_worst = schedule(campaign_h)

    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")
    _, _, _, Q_dot_worst = solve_steady_state(F_mesh, T_R_mesh, beta_worst, alpha, C_A0, T_in, C)
    feasible_mask = classify(Q_dot_worst, T_R_mesh) == FEASIBLE

    t = _time_samples(campaign_h, n)
    betas = np.array([schedule(ti)[1] for ti in t])

    best_total, best_F, best_T_R = -np.inf, None, None
    idx_F, idx_T = np.where(feasible_mask)
    for iF, iT in zip(idx_F, idx_T):
        F, T_R = F_mesh[iF, iT], T_R_mesh[iF, iT]
        _, C_b, _, Q_dot = solve_steady_state(F, T_R, betas, alpha, C_A0, T_in, C)
        profit = compute_profit(F, C_b, C_A0, Q_dot, price_B, price_A0, price_energy)
        total = float(np.trapezoid(profit, t))
        if total > best_total:
            best_total, best_F, best_T_R = total, float(F), float(T_R)

    return {"F": best_F, "T_R": best_T_R, "campaign_total_profit": best_total}


if __name__ == "__main__":
    import json
    import time

    from models.params import load_params
    from models.deactivation import make_severity_fn
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    econ = PARAMS["economics"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    prices = dict(price_B=econ["price_B"], price_A0=econ["price_A0"], price_energy=econ["price_energy"])

    CAMPAIGN_H = 75.0
    schedule = make_severity_fn("severe", CAMPAIGN_H, profile="exponential")

    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid = d["F"], d["T_R"]

    print("Finding best always-feasible fixed point (this may take a moment)...")
    best_fixed = find_best_always_feasible_point(F_grid, T_R_grid, schedule, CAMPAIGN_H, **kw, **prices)
    print(f"Best always-feasible fixed point: F={best_fixed['F']:.2f}, T_R={best_fixed['T_R']:.2f}, "
          f"campaign-total profit = {best_fixed['campaign_total_profit']:.1f}")

    adaptive_total = campaign_total_profit_adaptive(F_grid, T_R_grid, schedule, CAMPAIGN_H, **kw, **prices)
    print(f"Adaptive (continuously re-optimized) campaign-total profit = {adaptive_total:.1f}")

    gap_pct = 100.0 * (adaptive_total - best_fixed["campaign_total_profit"]) / adaptive_total
    print(f"\nTRUE INTEGRATED ADAPTATION GAP = {gap_pct:.2f}%")

    out_path = "open_loop_testing/campaign_integrated_gap.json"
    result = {"best_fixed": best_fixed, "adaptive_total": adaptive_total, "gap_pct": gap_pct,
              "campaign_h": CAMPAIGN_H}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"campaign_h": CAMPAIGN_H, "profile": "exponential", "severity": "severe", **prices},
        script=__file__, start_time=start_time, extra=result,
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")
