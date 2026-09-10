import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.params import load_params
from utils.plotstyle import (CONSTRAINT_COLOR, TEXT_SECONDARY, TIER_STYLE,
                             apply_style, save)

TRACES = "supervisors/campaign_traces_S3.json"
BEFORE, AFTER = 0.5, 2.5

P = load_params()
T_LIMIT = P["state_bounds"]["T_R"]["upper_soft"]
Q_FLOOR = P["input_bounds"]["Q_dot"]["lower"]

CASES = [("T0a_naive", "T0a_naive"), ("T0b_prudent", "T0b_prudent"),
         ("T2_rto", "T2_rto"), ("T3_median", "T3_rl")]


def main():
    t0 = time.time()
    apply_style()
    d = json.load(open(TRACES))
    C = d["cases"]
    step = d["step_at_h"]

    fig, (axT, axQ) = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)

    stats = {}
    for case, tier in CASES:
        st = TIER_STYLE[tier]
        c = C[case]
        t = np.asarray(c["t"]); T = np.asarray(c["T_R"]); Q = np.asarray(c["Q_dot"])
        w = (t >= step - BEFORE) & (t <= step + AFTER)
        kw = dict(color=st["color"], ls=st["ls"], lw=1.6, label=st["label"])
        axT.plot(t[w], T[w], **kw)
        axQ.plot(t[w], Q[w], **kw)

        ipk = int(np.argmax(T[w]))
        at_floor = Q[w] <= Q_FLOOR + 1e-6
        stats[case] = {"peak_T_R": float(T[w][ipk]),
                       "t_at_peak": float(t[w][ipk]),
                       "Q_at_peak": float(Q[w][ipk]),
                       "fraction_of_window_at_cooling_floor": float(at_floor.mean())}
        axT.plot([t[w][ipk]], [T[w][ipk]], marker=st["marker"], ms=6,
                 color=st["color"], zorder=5)

    axT.axhline(T_LIMIT, color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.3)
    axT.annotate(f"temperature limit {T_LIMIT:.0f} $\\degree$C",
                 xy=(0.99, T_LIMIT), xycoords=("axes fraction", "data"),
                 xytext=(0, 3), textcoords="offset points", ha="right",
                 va="bottom", fontsize=7.5, color=CONSTRAINT_COLOR)
    axQ.axhline(Q_FLOOR, color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.3)
    axQ.annotate(f"cooling floor {Q_FLOOR:.0f} kJ h$^{{-1}}$: no authority "
                 f"remains below this line",
                 xy=(0.99, Q_FLOOR), xycoords=("axes fraction", "data"),
                 xytext=(0, 4), textcoords="offset points", ha="right",
                 va="bottom", fontsize=7.5, color=CONSTRAINT_COLOR)

    for ax in (axT, axQ):
        ax.axvline(step, color=TEXT_SECONDARY, ls=":", lw=1.1, alpha=0.8)
    axT.annotate("$+0.6$ mol L$^{-1}$ feed step",
                 xy=(step, 0.30), xycoords=("data", "axes fraction"),
                 xytext=(5, 0), textcoords="offset points", ha="left",
                 va="center", fontsize=7.5, color=TEXT_SECONDARY)

    axT.set_ylabel("$T_R$ ($\\degree$C)")
    axQ.set_ylabel("$\\dot{Q}$ (kJ h$^{-1}$)")
    axQ.set_xlabel("campaign time (h)")
    axQ.set_xlim(step - BEFORE, step + AFTER)
    axT.legend(loc="center right", ncol=2, fontsize=7.5)

    fig.suptitle("Response to the held-out feed step at 30 h, severe ageing",
                 fontsize=10)
    fig.text(0.5, -0.03,
             "Markers show each tier's peak temperature. At that instant the "
             "cooling duty is at its floor for every tier except the prudent "
             "fixed target,\nwhose target sits far enough below the ceiling "
             "that it never needs the rest. The excursion is therefore limited "
             "by actuator\nauthority rather than by prediction, which is why "
             "Section 4.4 finds it insensitive to horizon length.",
             ha="center", va="top", fontsize=7.5, color=TEXT_SECONDARY)

    path = "supervisors/disturbance_zoom.png"
    save(fig, path,
         params={"traces": TRACES, "step_at_h": step, "window_h": [BEFORE, AFTER],
                 "T_limit": T_LIMIT, "Q_floor": Q_FLOOR,
                 "cases": [c for c, _ in CASES]},
         script=__file__, start_time=t0, extra=stats)

    print(f"{'case':14} {'peak T_R':>9} {'t':>7} {'Q at peak':>10} {'% at floor':>11}")
    for k, v in stats.items():
        print(f"{k:14} {v['peak_T_R']:9.3f} {v['t_at_peak']:7.2f} "
              f"{v['Q_at_peak']:10.1f} "
              f"{100*v['fraction_of_window_at_cooling_floor']:10.1f}%")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
