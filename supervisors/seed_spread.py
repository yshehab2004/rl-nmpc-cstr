import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plotstyle import (SURFACE, TEXT_SECONDARY, TIER_STYLE,
                             apply_style, save)

RESULTS = "supervisors/phase3_results_A1_new_withmargin.json"

SCENARIOS = [
    ("S1_no_ageing",     "S1\nno ageing"),
    ("S2_moderate",      "S2\nmoderate"),
    ("S3_severe",        "S3\nsevere"),
    ("OOD_beta010_100h", "OOD\n$\\beta\\to0.10$"),
    ("OOD_simultaneous", "OOD\nsimultaneous"),
]


def per_policy_means(runs):
    by_run = {}
    for r in runs:
        by_run.setdefault(r["rl_run"], []).append(r["profit_per_hour"])
    return {k: float(np.mean(v)) for k, v in sorted(by_run.items())}


def main():
    t0 = time.time()
    apply_style()
    data = json.load(open(RESULTS))

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(11.0, 4.4),
                             sharey=False)

    t3_style = TIER_STYLE["T3_rl"]
    t2_style = TIER_STYLE["T2_rto"]
    t0b_style = TIER_STYLE["T0b_prudent"]

    summary = {}
    for ax, (key, label) in zip(axes, SCENARIOS):
        tiers = data[key]["tiers"]
        policies = per_policy_means(tiers["T3_rl"]["runs"])
        vals = np.array(list(policies.values()))
        t2 = tiers["T2_rto"]["mean_profit_per_hour"]
        t0b = tiers["T0b_prudent"]["mean_profit_per_hour"]

        beats_t2 = vals > t2
        below_t0b = vals < t0b

        rng = np.random.default_rng(0)
        x = rng.uniform(-0.16, 0.16, size=len(vals))

        ax.axhline(t2, color=t2_style["color"], ls=t2_style["ls"], lw=1.5,
                   zorder=2)
        ax.axhline(t0b, color=t0b_style["color"], ls=t0b_style["ls"], lw=1.5,
                   zorder=2)

        ax.scatter(x[~beats_t2], vals[~beats_t2], s=38,
                   facecolors="none", edgecolors=t3_style["color"],
                   linewidths=1.3, zorder=4)
        ax.scatter(x[beats_t2], vals[beats_t2], s=52,
                   color=t3_style["color"], marker="D", zorder=5)

        n_beat, n_below = int(beats_t2.sum()), int(below_t0b.sum())
        ax.set_title(f"{label}\n{n_beat}/10 beat T2", fontsize=8.5)
        ax.set_xlim(-0.45, 0.45)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7.5)

        summary[key] = {"per_policy": policies, "T2": t2, "T0b": t0b,
                        "n_beating_T2": n_beat, "n_below_T0b": n_below,
                        "min": float(vals.min()), "max": float(vals.max()),
                        "spread_ratio": float(vals.max() / vals.min()),
                        "median": float(np.median(vals))}

    axes[0].set_ylabel("profit per hour")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="D", ls="none", color=t3_style["color"],
               markersize=6, label="learned policy, beats T2"),
        Line2D([], [], marker="o", ls="none", mfc="none",
               mec=t3_style["color"], markersize=6, label="learned policy"),
        Line2D([], [], color=t2_style["color"], ls=t2_style["ls"], lw=1.5,
               label="T2 classical optimiser"),
        Line2D([], [], color=t0b_style["color"], ls=t0b_style["ls"], lw=1.5,
               label="T0b prudent fixed target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Ten independently trained policies per scenario, differing "
                 "only in training seed", fontsize=10)
    fig.text(0.5, -0.135,
             "Each point is one policy averaged over five campaign seeds. "
             "Axes are per-scenario: the scenarios differ in achievable "
             "profit, so levels are not\ncomparable between panels, only the "
             "spread within each and its position against the two "
             "comparators.",
             ha="center", va="top", fontsize=7.5, color=TEXT_SECONDARY)

    path = "supervisors/seed_spread.png"
    save(fig, path,
         params={"results": RESULTS,
                 "scenarios": [k for k, _ in SCENARIOS],
                 "policies_per_scenario": 10, "campaign_seeds": 5,
                 "jitter_seed": 0},
         script=__file__, start_time=t0, extra=summary)

    for k, _ in SCENARIOS:
        s = summary[k]
        print(f"{k:20} spread {s['min']:7.1f}-{s['max']:7.1f} "
              f"({s['spread_ratio']:.2f}x)  T2={s['T2']:7.1f}  "
              f"beat={s['n_beating_T2']}/10  belowT0b={s['n_below_T0b']}/10")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
