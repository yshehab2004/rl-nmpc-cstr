import json
import time

import numpy as np

from controllers.nmpc import build_nmpc, run_closed_loop
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from open_loop_testing.optimum_trajectory import compute_profit
from open_loop_testing.target_optimizer import solve_target
from simulation.simulator import build_simulator, initial_state

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]
ECON = PARAMS["economics"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])
SAFETY_W = ECON["safety_penalty_weight"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]

BETAS = [1.0, 0.6, 0.4, 0.3, 0.2]
MARGINS = [0.0, 1.0, 2.0, 3.0]
N_STEPS = 100


def evaluate(beta, margin):
    projected = solve_target(35.0, 133.0, margin=margin, beta_hat=beta,
                              C_A0=FEED["C_A0"], alpha=1.0, T_in=FEED["T_in"], C=C)
    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(base=FEED["C_A0"], magnitude=0.6, t_start=0.1)
    )
    mpc = build_nmpc(target=(projected["F_s"], projected["T_R_s"]),
                      n_horizon=20, beta_belief=beta)
    simulator = build_simulator(ageing_fn=lambda t_now: (1.0, beta),
                                 disturbance_fn=disturbed)
    run = run_closed_loop(mpc, simulator, initial_state(), n_steps=N_STEPS)

    T_R = run["x"][:, 2]
    C_b = run["x"][1:, 1]
    F = run["u"][:, 0]
    Q_dot = run["u"][:, 1]

    C_A0_actual = np.array([disturbed(t)[0] for t in run["t"][1:]])
    profit = compute_profit(F, C_b, C_A0_actual, Q_dot, **PRICES)

    violation = np.maximum(0.0, T_R - T_LIMIT)
    saturated = float(np.mean(Q_dot <= -8500.0 + 1e-3))

    return {
        "beta": beta, "margin": margin,
        "target_F": projected["F_s"], "target_T_R": projected["T_R_s"],
        "achieved_F_final": float(F[-1]),
        "T_R_max": float(T_R.max()),
        "violation_max": float(violation.max()),
        "violation_integral": float(violation.sum()),
        "steps_above_limit": int((T_R > T_LIMIT).sum()),
        "mean_profit": float(np.mean(profit)),
        "saturated_fraction": saturated,
    }


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    results = [evaluate(b, m) for b in BETAS for m in MARGINS]

    print(f"{'beta':>5} {'margin':>7} {'targetT_R':>10} {'T_R max':>8} "
          f"{'viol max':>9} {'viol int':>9} {'mean profit':>12} {'sat frac':>9}")
    for r in results:
        print(f"{r['beta']:5.2f} {r['margin']:7.1f} {r['target_T_R']:10.2f} "
              f"{r['T_R_max']:8.3f} {r['violation_max']:9.4f} "
              f"{r['violation_integral']:9.3f} {r['mean_profit']:12.2f} "
              f"{r['saturated_fraction']:9.2f}")

    print(f"\nExpected cost = -mean_profit + {SAFETY_W} * violation_integral / n_steps")
    print(f"{'beta':>5} " + " ".join(f"{'m=' + str(m):>12}" for m in MARGINS) + "   best")
    summary = {}
    for beta in BETAS:
        rows = [r for r in results if r["beta"] == beta]
        costs = {r["margin"]: -r["mean_profit"] + SAFETY_W * r["violation_integral"] / N_STEPS
                 for r in rows}
        best = min(costs, key=costs.get)
        summary[beta] = {"costs": costs, "best_margin": best}
        print(f"{beta:5.2f} " + " ".join(f"{costs[m]:12.2f}" for m in MARGINS)
              + f"   {best:.1f}")

    out_path = "controllers/margin_derivation.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "summary": {str(k): v for k, v in summary.items()}},
                   f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"betas": BETAS, "margins": MARGINS, "n_steps": N_STEPS,
                "safety_penalty_weight": SAFETY_W, "C_A0_step": 0.6, **PRICES},
        script=__file__, start_time=start_time,
        extra={"results": results, "summary": {str(k): v for k, v in summary.items()}},
    )
    print(f"\nSaved {out_path}, {sidecar}")
