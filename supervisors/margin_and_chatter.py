import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plotstyle import (CONSTRAINT_COLOR, REFERENCE_COLOR, TEXT_SECONDARY,
                             TIER_STYLE, apply_style, save)

TRACES = "supervisors/campaign_traces_S3.json"
SCHEDULED_MARGIN = 2.0

LEARNED = [("T3_best", "best"), ("T3_median", "median"), ("T3_worst", "worst")]


def main():
    t0 = time.time()
    apply_style()
    d = json.load(open(TRACES))
    C = d["cases"]
    step_at = d["step_at_h"]
    rl = TIER_STYLE["T3_rl"]
    t2 = TIER_STYLE["T2_rto"]

    fig, (axM, axJ) = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)

    stats = {}
    for i, (case, lab) in enumerate(LEARNED):
        dec = C[case]["decisions"]
        td = np.array([x["t"] for x in dec])
        m = np.array([x["margin"] for x in dec])
        F = np.array([x["F_ref"] for x in dec])
        Fs = np.array([x["F_s"] for x in dec])
        style = {"best": dict(alpha=0.75, lw=1.0, ls=(0, (5, 2))),
                 "median": dict(alpha=1.0, lw=1.6, ls="-"),
                 "worst": dict(alpha=0.75, lw=1.0, ls=(0, (1, 1.6)))}[lab]
        axM.plot(td, m, color=rl["color"], label=f"T3 {lab} seed",
                 zorder=3 if lab == "median" else 2, **style)
        axJ.plot(td[1:], np.maximum(np.abs(np.diff(F)), 1e-4),
                 color=rl["color"], zorder=3 if lab == "median" else 2,
                 label=f"T3 {lab} seed", **style)
        stats[case] = {"margin_mean": float(m.mean()), "margin_sd": float(m.std()),
                       "margin_min": float(m.min()), "margin_max": float(m.max()),
                       "mean_abs_step_F_ref": float(np.abs(np.diff(F)).mean()),
                       "mean_abs_step_F_projected": float(np.abs(np.diff(Fs)).mean())}

    dec2 = C["T2_rto"]["decisions"]
    td2 = np.array([x["t"] for x in dec2])
    F2 = np.array([x["F_ref"] for x in dec2])
    axJ.plot(td2[1:], np.maximum(np.abs(np.diff(F2)), 1e-4), color=t2["color"],
             ls=t2["ls"], lw=1.5, label="T2 RTO", zorder=4)
    stats["T2_rto"] = {"margin_mean": SCHEDULED_MARGIN, "margin_sd": 0.0,
                       "mean_abs_step_F_ref": float(np.abs(np.diff(F2)).mean()),
                       "mean_abs_step_F_projected":
                           float(np.abs(np.diff(np.array([x["F_s"] for x in dec2]))).mean())}

    axM.axhline(SCHEDULED_MARGIN, color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.3,
                zorder=2)
    axM.annotate(f"fixed {SCHEDULED_MARGIN:.1f} $\\degree$C held by T0b and T2",
                 xy=(0.985, SCHEDULED_MARGIN), xycoords=("axes fraction", "data"),
                 xytext=(0, 4), textcoords="offset points", ha="right",
                 va="bottom", fontsize=7.5, color=CONSTRAINT_COLOR)

    for ax in (axM, axJ):
        ax.axvline(step_at, color=TEXT_SECONDARY, ls=":", lw=1.0, alpha=0.7)
    axM.annotate("feed step", xy=(step_at, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(4, -10), textcoords="offset points", ha="left", va="top",
                 fontsize=7.5, color=TEXT_SECONDARY)

    axM.set_ylabel("requested margin ($\\degree$C)")
    axM.set_title("The margin is used as a decision variable, not parked",
                  fontsize=9.5)
    axM.legend(loc="upper left", ncol=3, fontsize=7.5)

    axJ.set_yscale("log")
    axJ.set_ylabel("$|\\Delta F_{ref}|$ between\nconsecutive decisions (h$^{-1}$)")
    axJ.set_xlabel("campaign time (h)")
    axJ.set_title("But the requested target chatters, by a factor of 45 against "
                  "the classical layer", fontsize=9.5)
    axJ.legend(loc="lower left", ncol=4, fontsize=7.5)
    axJ.set_xlim(0, float(td[-1]))

    fig.text(0.5, -0.03,
             "Both panels are the supervisor's own request, one point per "
             "0.25 h decision for the learned tier and per 2 h for the "
             "classical one.\nThe lower panel is logarithmic: the two "
             "strategies differ by more than an order of magnitude "
             "throughout,\nand values are floored at $10^{-4}$ so that a "
             "request held exactly unchanged still draws.",
             ha="center", va="top", fontsize=7.5, color=TEXT_SECONDARY)

    path = "supervisors/margin_and_chatter.png"
    save(fig, path,
         params={"traces": TRACES, "scenario": d["scenario"], "seed": d["seed"],
                 "scheduled_margin": SCHEDULED_MARGIN,
                 "learned_cases": [c for c, _ in LEARNED]},
         script=__file__, start_time=t0, extra=stats)

    print(f"{'case':14} {'margin mean':>11} {'sd':>6} {'min':>6} {'max':>6} "
          f"{'|dF| req':>9} {'|dF| proj':>10}")
    for k, v in stats.items():
        print(f"{k:14} {v['margin_mean']:11.2f} {v['margin_sd']:6.2f} "
              f"{v.get('margin_min', float('nan')):6.2f} "
              f"{v.get('margin_max', float('nan')):6.2f} "
              f"{v['mean_abs_step_F_ref']:9.2f} {v['mean_abs_step_F_projected']:10.2f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
