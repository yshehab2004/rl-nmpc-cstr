import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.params import load_params
from utils.plotstyle import (CONSTRAINT_COLOR, SURFACE, TEXT_SECONDARY,
                             TIER_STYLE, TRUTH_COLOR, apply_style, save)

TRACES = "supervisors/campaign_traces_S3.json"

P = load_params()
T_LIMIT = P["state_bounds"]["T_R"]["upper_soft"]
Q_FLOOR = P["input_bounds"]["Q_dot"]["lower"]

MAIN = [("T0a_naive", "T0a_naive"), ("T0b_prudent", "T0b_prudent"),
        ("T2_rto", "T2_rto"), ("T3_median", "T3_rl")]

T_R_YLIM = (128.0, 141.5)


def main():
    t0 = time.time()
    apply_style()
    d = json.load(open(TRACES))
    C = d["cases"]
    step_at = d["step_at_h"]

    fig, axes = plt.subplots(5, 1, figsize=(9.2, 11.0), sharex=True,
                             height_ratios=[1.25, 1, 1, 1, 1])
    axT, axF, axQ, axCb, axB = axes

    t = np.asarray(C["T3_median"]["t"])

    for case, tier in MAIN:
        st = TIER_STYLE[tier]
        c = C[case]
        tt = np.asarray(c["t"])
        kw = dict(color=st["color"], ls=st["ls"], lw=1.5, label=st["label"],
                  zorder=3)
        axT.plot(tt, c["T_R"], **kw)
        axF.plot(tt, c["F"], **kw)
        axQ.plot(tt, np.abs(np.asarray(c["Q_dot"])), **kw)
        axCb.plot(tt, c["C_b"], **kw)

    axT.axhline(T_LIMIT, color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.3, zorder=2)
    axT.annotate(f"temperature limit {T_LIMIT:.0f} $\\degree$C",
                 xy=(0.5, T_LIMIT), xycoords=("axes fraction", "data"),
                 xytext=(0, -3), textcoords="offset points", ha="center",
                 va="top", fontsize=7.5, color=CONSTRAINT_COLOR)
    axQ.axhline(abs(Q_FLOOR), color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.3, zorder=2)
    axQ.annotate(f"cooling capacity {abs(Q_FLOOR):.0f} kJ h$^{{-1}}$",
                 xy=(0.015, 0.06), xycoords="axes fraction", ha="left",
                 va="bottom", fontsize=7.5, color=CONSTRAINT_COLOR)

    axB.plot(t, C["T2_rto"]["beta_true"], color=TRUTH_COLOR, lw=1.6,
             label="$\\beta$ true", zorder=3)
    axB.plot(t, C["T2_rto"]["beta_hat"], color=TIER_STYLE["T2_rto"]["color"],
             ls="-.", lw=1.4, label="$\\hat{\\beta}$ estimated", zorder=3)

    for ax in axes:
        ax.axvline(step_at, color=TEXT_SECONDARY, ls=":", lw=1.0, alpha=0.7,
                   zorder=1.5)
    axT.annotate("feed step", xy=(step_at, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(4, -10), textcoords="offset points", ha="left",
                 va="top", fontsize=7.5, color=TEXT_SECONDARY)

    axT.set_ylim(*T_R_YLIM)
    axT.set_ylabel("$T_R$ ($\\degree$C)")
    axF.set_ylabel("$F$ (h$^{-1}$)")
    axQ.set_ylabel("$|\\dot{Q}|$ (kJ h$^{-1}$)")
    axCb.set_ylabel("$C_B$ (mol L$^{-1}$)")
    axB.set_ylabel("catalyst activity")
    axB.set_xlabel("campaign time (h)")
    axB.set_xlim(0, float(t[-1]))

    axT.legend(loc="lower left", ncol=4, fontsize=7.5)
    axB.legend(loc="upper right", ncol=2, fontsize=7.5)

    fig.suptitle("One 75 h campaign under severe ageing, all four supervisory "
                 "strategies", fontsize=10)
    fig.text(0.5, -0.012,
             "The learned trace is the median of ten seeds. Its temperature "
             "panel is clipped at 128 $\\degree$C; its start-up transient runs "
             "below that.",
             ha="center", va="top", fontsize=7.5, color=TEXT_SECONDARY)

    summary = {k: {"profit_per_hour": v["metrics"]["total_profit"] / 75.0,
                   "T_R_max": v["metrics"]["T_R_max"],
                   "fraction_time_violating": v["metrics"]["fraction_time_violating"]}
               for k, v in C.items()}
    path = "supervisors/campaign_trajectory.png"
    save(fig, path,
         params={"traces": TRACES, "scenario": d["scenario"], "seed": d["seed"],
                 "beta_final": d["beta_final"], "step_at_h": step_at,
                 "T_limit": T_LIMIT, "Q_floor": Q_FLOOR,
                 "lines": [m[0] for m in MAIN], "T_R_ylim": list(T_R_YLIM)},
         script=__file__, start_time=t0, extra=summary)
    for k, v in summary.items():
        print(f"  {k:14} profit/h={v['profit_per_hour']:8.1f} "
              f"T_R_max={v['T_R_max']:7.3f} frac={v['fraction_time_violating']:.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
