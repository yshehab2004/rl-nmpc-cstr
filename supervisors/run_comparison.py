import json
import time

from models.deactivation import make_deactivation_fn
from models.disturbances import (build_feed_disturbance, ramp_disturbance,
                                  step_disturbance)
from models.params import load_params
from supervisors.campaign import run_campaign
from supervisors.tiers import RTOSupervisor, T0a_naive, T0b_prudent

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]

SEED = 20260905
CAMPAIGN_H = 75.0
SHORT_H = 20.0
C_A0_MAG = 0.6
T_IN_MAG = 3.0


def _c_a0_step(horizon):
    return build_feed_disturbance(
        C_A0_gen=step_disturbance(FEED["C_A0"], C_A0_MAG, 0.4 * horizon))


def _c_a0_ramp(horizon):
    return build_feed_disturbance(
        C_A0_gen=ramp_disturbance(FEED["C_A0"], C_A0_MAG,
                                   0.25 * horizon, 0.65 * horizon))


def _t_in_step(horizon):
    return build_feed_disturbance(
        T_in_gen=step_disturbance(FEED["T_in"], T_IN_MAG, 0.4 * horizon))


def scenarios():
    severe = make_deactivation_fn(beta_final=0.20, campaign_h=CAMPAIGN_H)
    moderate = make_deactivation_fn(beta_final=0.35, campaign_h=CAMPAIGN_H)
    return [
        ("E1_nominal",         SHORT_H,    None,     None),
        ("E2_C_A0_step",       SHORT_H,    None,     _c_a0_step(SHORT_H)),
        ("E3_C_A0_ramp",       SHORT_H,    None,     _c_a0_ramp(SHORT_H)),
        ("E3b_T_in_step",      SHORT_H,    None,     _t_in_step(SHORT_H)),
        ("E4a_ageing_moderate", CAMPAIGN_H, moderate, None),
        ("E4b_ageing_severe",  CAMPAIGN_H, severe,   None),
        ("E5_ageing_plus_step", CAMPAIGN_H, severe,  _c_a0_step(CAMPAIGN_H)),
    ]


def tiers():
    return [
        ("T0a_naive", T0a_naive),
        ("T0b_prudent", T0b_prudent),
        ("T2_rto", lambda: RTOSupervisor(cadence_h=2.0)),
    ]


def main(only=None):
    results = {}
    for name, horizon, ageing_fn, disturbance_fn in scenarios():
        if only and name not in only:
            continue
        results[name] = {"campaign_h": horizon, "tiers": {}}
        print(f"\n=== {name}  ({horizon:.0f}h) ===", flush=True)
        for tier_name, factory in tiers():
            t0 = time.time()
            run = run_campaign(factory(), campaign_h=horizon, ageing_fn=ageing_fn,
                                disturbance_fn=disturbance_fn, seed=SEED, n_horizon=20)
            m = dict(run["metrics"])
            m["wall_time_s"] = time.time() - t0
            m["n_supervisor_solves"] = run["n_supervisor_solves"]
            m["profit_per_hour"] = m["total_profit"] / horizon
            results[name]["tiers"][tier_name] = m
            print(f"  {tier_name:14} profit/h={m['profit_per_hour']:10.1f}  "
                  f"viol={m['violation_integral_degC_h']:8.3f} degC.h  "
                  f"viol_max={m['violation_max']:6.3f}  "
                  f"T_R_max={m['T_R_max']:7.2f}  "
                  f"beta_err={m['beta_hat_max_error']:.3f}  "
                  f"solves={m['n_supervisor_solves']:4d}  ({m['wall_time_s']:.0f}s)",
                  flush=True)
    return results


def report(results):
    names = [n for n, _ in tiers()]
    print("\n" + "=" * 86)
    print("PROFIT PER HOUR (higher better)  /  VIOLATION degC.h (lower better)")
    print("=" * 86)
    print(f"{'scenario':>20} " + " ".join(f"{n:>20}" for n in names))
    for scenario, row in results.items():
        cells = []
        for n in names:
            m = row["tiers"].get(n)
            cells.append("--".rjust(20) if m is None else
                          f"{m['profit_per_hour']:10.0f}/{m['violation_integral_degC_h']:8.2f}")
        print(f"{scenario:>20} " + " ".join(cells))


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    results = main()
    report(results)

    out_path = "supervisors/tier_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"seed": SEED, "campaign_h": CAMPAIGN_H, "short_h": SHORT_H,
                "decision_interval_h": PARAMS["rl"]["decision_interval_h"],
                "rto_cadence_h": 2.0, "n_horizon": 20,
                "C_A0_magnitude": C_A0_MAG, "T_in_magnitude": T_IN_MAG,
                "beta_final": {"moderate": 0.35, "severe": 0.20},
                "scenarios": [s[0] for s in scenarios()]},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"\nTotal wall time: {time.time() - start_time:.0f}s")
    print(f"Saved {out_path}, {sidecar}")
