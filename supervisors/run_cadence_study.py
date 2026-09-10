import json
import time

import numpy as np

from models.deactivation import make_deactivation_fn
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from supervisors.campaign import run_campaign
from supervisors.tiers import RTOSupervisor
from utils import plotstyle as ps

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]

SEED = 20260906
CAMPAIGN_H = 20.0
C_A0_MAG = 0.6
CADENCES = (2.0, 0.25)


def scenarios():
    return [
        ("mild_no_ageing", None),
        ("severe_ageing", make_deactivation_fn(beta_final=0.25, campaign_h=CAMPAIGN_H)),
    ]


def main():
    disturbance = build_feed_disturbance(
        C_A0_gen=step_disturbance(FEED["C_A0"], C_A0_MAG, 0.4 * CAMPAIGN_H))
    results = {}

    for name, ageing_fn in scenarios():
        results[name] = {}
        print(f"\n=== {name} ===", flush=True)
        for cadence in CADENCES:
            t0 = time.time()
            run = run_campaign(RTOSupervisor(cadence_h=cadence),
                                campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                                disturbance_fn=disturbance, seed=SEED,
                                n_horizon=20)
            m = dict(run["metrics"])
            m["profit_per_hour"] = m["total_profit"] / CAMPAIGN_H
            m["n_solves"] = int(run["n_supervisor_solves"])
            m["wall_time_s"] = time.time() - t0
            m["F_s_std"] = float(np.std(run["F_s"]))
            m["F_s_reversals"] = float(np.mean(
                np.sign(np.diff(run["F_s"][1:])) * np.sign(np.diff(run["F_s"][:-1])) < 0))
            results[name][f"cadence_{cadence}h"] = m
            print(f"  cadence {cadence:5.2f}h  profit/h={m['profit_per_hour']:9.1f}  "
                  f"solves={m['n_solves']:4d}  viol={m['violation_integral_degC_h']:7.4f}  "
                  f"F_s std={m['F_s_std']:.3f}  ({m['wall_time_s']:.0f}s)", flush=True)

        slow = results[name]["cadence_2.0h"]["profit_per_hour"]
        fast = results[name]["cadence_0.25h"]["profit_per_hour"]
        delta = 100.0 * (fast - slow) / slow
        results[name]["fast_vs_slow_pct"] = delta
        verdict = ("fast is WORSE" if delta < -0.1 else
                   "fast is better" if delta > 0.1 else "no difference")
        print(f"  -> 0.25h vs 2h: {delta:+.2f}%  ({verdict})", flush=True)

    return results


def plot(results, path, script, start_time=None):
    import matplotlib.pyplot as plt

    ps.apply_style()
    names = [n for n, _ in scenarios()]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    colors = {2.0: ps.tier("T2_rto")["color"], 0.25: ps.tier("T0b_prudent")["color"]}
    markers = {2.0: "^", 0.25: "s"}

    for ax, metric, ylabel, title in (
            (axes[0], "profit_per_hour", "profit (currency h$^{-1}$)", "outcome"),
            (axes[1], "F_s_std", "std of commanded F (h$^{-1}$)", "mechanism: target jitter")):
        for cadence in CADENCES:
            ys = [results[n][f"cadence_{cadence}h"][metric] for n in names]
            ax.plot(range(len(names)), ys, ls="none", marker=markers[cadence],
                    color=colors[cadence], ms=9, label=f"RTO every {cadence:g} h")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, color=ps.TEXT_SECONDARY, loc="left")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.25 * (hi - lo))
    axes[0].legend(loc="best")

    fig.suptitle("Solving more often does not help an RTO that reads only a noisy estimate",
                  fontsize=10)
    return ps.save(fig, path, {"seed": SEED, "campaign_h": CAMPAIGN_H}, script,
                    start_time=start_time)


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    results = main()

    plot(results, "supervisors/rto_cadence.png", __file__, start_time=start_time)

    out_path = "supervisors/cadence_study.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"seed": SEED, "campaign_h": CAMPAIGN_H, "cadences_h": list(CADENCES),
                "C_A0_magnitude": C_A0_MAG, "n_horizon": 20,
                "scenarios": [n for n, _ in scenarios()]},
        script=__file__, start_time=start_time, extra={"results": results})
    print(f"\nSaved {out_path}, {sidecar}")
