import json
import time

import numpy as np

from models.deactivation import make_deactivation_fn
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from supervisors.campaign import run_campaign
from supervisors.tiers import FixedSupervisor, RTOSupervisor

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
Q_LOWER = PARAMS["input_bounds"]["Q_dot"]["lower"]
F_UPPER = PARAMS["input_bounds"]["F"]["upper"]
SENSOR_SIGMA = PARAMS["measurement"]["noise_frac"] * PARAMS["measurement"]["T_R_range"]

SEED = 20260906
CAMPAIGN_H = 6.0
STEP_AT_H = 2.0
PUBLISHED_MAX = 5.7

BETAS = (0.9, 0.6, 0.45)
MAGNITUDES = (0.6, 1.2, 1.8, 2.4, 3.0)
MARGINS = (0.0, 2.5)


def _rto(margin):
    return RTOSupervisor(cadence_h=0.25, margin=margin, uncertainty=0.0)


def one_case(beta, magnitude, margin):
    ageing_fn = make_deactivation_fn(beta_final=beta, campaign_h=1e-6)
    disturbance = build_feed_disturbance(
        C_A0_gen=step_disturbance(FEED["C_A0"], magnitude, STEP_AT_H))
    run = run_campaign(_rto(margin), campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                        disturbance_fn=disturbance, seed=SEED, n_horizon=20,
                        use_estimator=False, decision_interval_h=0.25)
    T_R, F, Q = run["x"][:, 2], run["u"][:, 0], run["u"][:, 1]
    after = run["t"] >= STEP_AT_H
    return {
        "peak_excursion": float(max(0.0, T_R[after].max() - T_LIMIT)),
        "violation_integral": float(run["metrics"]["violation_integral_degC_h"]),
        "profit_per_hour": float(run["metrics"]["total_profit"] / CAMPAIGN_H),
        "frac_F_at_bound": float(np.mean(F[after] >= F_UPPER - 1e-3)),
        "frac_cooling_saturated": float(np.mean(Q[after] <= Q_LOWER + 1.0)),
        "C_A0_final": FEED["C_A0"] + magnitude,
    }


def main():
    results = {}
    for beta in BETAS:
        for magnitude in MAGNITUDES:
            key = f"beta{beta}_step{magnitude}"
            results[key] = {"beta": beta, "magnitude": magnitude,
                             "inside_published_range":
                                 FEED["C_A0"] + magnitude <= PUBLISHED_MAX,
                             "margins": {}}
            for margin in MARGINS:
                t0 = time.time()
                m = one_case(beta, magnitude, margin)
                m["wall_s"] = time.time() - t0
                results[key]["margins"][f"margin_{margin}"] = m
            zero = results[key]["margins"]["margin_0.0"]
            held = results[key]["margins"]["margin_2.5"]
            results[key]["excursion_prevented"] = (
                zero["peak_excursion"] - held["peak_excursion"])
            results[key]["profit_given_up"] = (
                zero["profit_per_hour"] - held["profit_per_hour"])
            flag = "" if results[key]["inside_published_range"] else "  [OUTSIDE PUBLISHED RANGE]"
            print(f"beta={beta:.2f} step=+{magnitude:.1f} "
                  f"(C_A0={m['C_A0_final']:.1f}){flag}", flush=True)
            print(f"   margin 0.0: peak +{zero['peak_excursion']:.3f} degC  "
                  f"F@bound {100*zero['frac_F_at_bound']:.0f}%  "
                  f"cool_sat {100*zero['frac_cooling_saturated']:.0f}%  "
                  f"profit/h {zero['profit_per_hour']:.0f}", flush=True)
            print(f"   margin 2.5: peak +{held['peak_excursion']:.3f} degC  "
                  f"F@bound {100*held['frac_F_at_bound']:.0f}%  "
                  f"cool_sat {100*held['frac_cooling_saturated']:.0f}%  "
                  f"profit/h {held['profit_per_hour']:.0f}", flush=True)
            print(f"   -> excursion prevented {results[key]['excursion_prevented']:+.3f} degC "
                  f"(sensor sigma {SENSOR_SIGMA:.2f}), "
                  f"profit given up {results[key]['profit_given_up']:+.0f}/h",
                  flush=True)
    return results


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start = time.time()
    results = main()

    print("\n" + "=" * 78)
    print("DOES HEADROOM PAY?  An excursion only counts if it exceeds the")
    print(f"sensor's own noise ({SENSOR_SIGMA:.2f} degC); below that nothing")
    print("on a real plant could tell the two policies apart.")
    print("=" * 78)
    pays = [k for k, v in results.items()
            if v["excursion_prevented"] > SENSOR_SIGMA]
    pays_in_range = [k for k in pays if results[k]["inside_published_range"]]
    print(f"cases where margin prevents a DETECTABLE excursion: "
          f"{len(pays)} of {len(results)}")
    print(f"  ... of those, inside the published C_A0 range: {len(pays_in_range)}")
    if not pays_in_range:
        print("\n  Within this plant's real feed envelope, cooling headroom buys")
        print("  nothing measurable. The SSTO absorbs the disturbance first.")
        print("  That is a finding about the thesis, not a failed experiment.")
    else:
        for k in pays_in_range:
            v = results[k]
            print(f"  {k}: prevents {v['excursion_prevented']:.2f} degC "
                  f"for {v['profit_given_up']:.0f}/h")

    out = "supervisors/headroom_value.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None,
        params={"betas": list(BETAS), "magnitudes": list(MAGNITUDES),
                "margins": list(MARGINS), "campaign_h": CAMPAIGN_H,
                "step_at_h": STEP_AT_H, "seed": SEED,
                "use_estimator": False, "decision_interval_h": 0.25},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")
