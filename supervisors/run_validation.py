import json
import time

import numpy as np

from models.deactivation import make_deactivation_fn
from models.disturbances import (build_feed_disturbance, ramp_disturbance,
                                  step_disturbance)
from models.params import load_params
from supervisors.campaign import run_campaign
from supervisors.plots import plot_campaign, plot_tier_comparison
from supervisors.tiers import RTOSupervisor, T0a_naive, T0b_prudent

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
Q_LOWER, Q_UPPER = PARAMS["input_bounds"]["Q_dot"]["lower"], PARAMS["input_bounds"]["Q_dot"]["upper"]
F_LO, F_HI = PARAMS["input_bounds"]["F"]["lower"], PARAMS["input_bounds"]["F"]["upper"]

SEED = 20260905
SOLVER_TOL = 1e-3
SHORT_H, LONG_H = 20.0, 75.0
C_A0_MAG, T_IN_MAG = 0.6, 3.0


def scenarios():
    severe = make_deactivation_fn(beta_final=0.20, campaign_h=LONG_H)
    return [
        ("V1_C_A0_step", SHORT_H, None,
         build_feed_disturbance(C_A0_gen=step_disturbance(FEED["C_A0"], C_A0_MAG, 0.4 * SHORT_H)),
         "C$_{A0}$ step +0.6 at 8 h, no ageing"),
        ("V2_C_A0_ramp", SHORT_H, None,
         build_feed_disturbance(C_A0_gen=ramp_disturbance(FEED["C_A0"], C_A0_MAG, 8.0, 13.0)),
         "C$_{A0}$ ramp +0.6 over 8-13 h, no ageing"),
        ("V3_T_in_step", SHORT_H, None,
         build_feed_disturbance(T_in_gen=step_disturbance(FEED["T_in"], T_IN_MAG, 0.4 * SHORT_H)),
         "T$_{in}$ step +3$\\,^\\circ$C at 8 h, no ageing"),
        ("V4_step_under_ageing", LONG_H, severe,
         build_feed_disturbance(C_A0_gen=step_disturbance(FEED["C_A0"], C_A0_MAG, 0.4 * LONG_H)),
         "C$_{A0}$ step +0.6 at 30 h, severe ageing"),
    ]


def _sign_change_fraction(signal, skip=50):
    d = np.diff(signal[skip:])
    if len(d) < 3:
        return 0.0
    return float(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0) / (len(d) - 1))


def check(run):
    x, u = run["x"], run["u"]
    T_R, F, Q_dot = x[:, 2], u[:, 0], u[:, 1]

    checks = {
        "finite": bool(np.all(np.isfinite(x)) and np.all(np.isfinite(u))),
        "T_R_bounded": bool(T_R.min() > 50.0 and T_R.max() < 200.0),
        "F_in_bounds": bool(F.min() >= F_LO - SOLVER_TOL and F.max() <= F_HI + SOLVER_TOL),
        "Q_dot_in_bounds": bool(Q_dot.min() >= Q_LOWER - SOLVER_TOL
                                 and Q_dot.max() <= Q_UPPER + SOLVER_TOL),
        "not_oscillating": bool(_sign_change_fraction(F) < 0.25),
        "violation_bounded": bool(max(0.0, T_R.max() - T_LIMIT) < 10.0),
    }
    return all(checks.values()), checks


def main():
    tiers = [("T0a_naive", T0a_naive), ("T0b_prudent", T0b_prudent),
             ("T2_rto", lambda: RTOSupervisor(cadence_h=2.0))]
    results, all_passed = {}, True

    for name, horizon, ageing_fn, disturbance_fn, caption in scenarios():
        results[name] = {"campaign_h": horizon, "tiers": {}}
        print(f"\n=== {name}  ({horizon:.0f}h) ===", flush=True)
        for tier_name, factory in tiers:
            t0 = time.time()
            run = run_campaign(factory(), campaign_h=horizon, ageing_fn=ageing_fn,
                                disturbance_fn=disturbance_fn, seed=SEED, n_horizon=20)
            passed, checks = check(run)
            all_passed &= passed
            m = dict(run["metrics"])
            m.update(passed=passed, checks=checks, wall_time_s=time.time() - t0,
                      profit_per_hour=m["total_profit"] / horizon)
            results[name]["tiers"][tier_name] = m

            fig_path = f"supervisors/{name}_{tier_name}.png"
            plot_campaign(run, tier_name, f"{name}: {caption}", fig_path,
                           script=__file__,
                           params={"scenario": name, "tier": tier_name,
                                   "campaign_h": horizon, "seed": SEED})

            flag = "PASS" if passed else "FAIL " + ",".join(k for k, v in checks.items() if not v)
            print(f"  {tier_name:14} profit/h={m['profit_per_hour']:9.1f}  "
                  f"viol={m['violation_integral_degC_h']:8.3f} degC.h  "
                  f"T_R_max={m['T_R_max']:7.2f}  "
                  f"beta_err={m['beta_hat_max_error']:.3f}  "
                  f"[{flag}]  ({m['wall_time_s']:.0f}s)", flush=True)

    return results, all_passed


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    results, all_passed = main()

    print("\n" + "=" * 70)
    print("PHASE 1 VALIDATION: " + ("ALL PASS" if all_passed else "FAILURES PRESENT"))
    print("=" * 70)

    plot_tier_comparison(results, "supervisors/validation_summary_two_axis.png",
                          script=__file__, start_time=start_time,
                          params={"seed": SEED, "scenarios": list(results)})

    out_path = "supervisors/validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"seed": SEED, "short_h": SHORT_H, "long_h": LONG_H,
                "C_A0_magnitude": C_A0_MAG, "T_in_magnitude": T_IN_MAG,
                "rto_cadence_h": 2.0, "n_horizon": 20,
                "scenarios": [s[0] for s in scenarios()]},
        script=__file__, start_time=start_time,
        extra={"results": results, "all_passed": all_passed})
    print(f"Saved {out_path}, {sidecar}")
    print(f"Total wall time: {time.time() - start_time:.0f}s")
