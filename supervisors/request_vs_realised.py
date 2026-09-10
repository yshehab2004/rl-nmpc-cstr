import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.params import load_params
from matplotlib.colors import LinearSegmentedColormap
from utils.plotstyle import (BETA_CMAP, CONSTRAINT_COLOR, TEXT_SECONDARY,
                             apply_style, save)

TRACES = "supervisors/campaign_traces_S3.json"
PANELS = [("T2_rto", "T2 classical optimiser"),
          ("T3_median", "T3 learned, median seed")]

P = load_params()
T_LIMIT = P["state_bounds"]["T_R"]["upper_soft"]
F_LO, F_HI = P["input_bounds"]["F"]["lower"], P["input_bounds"]["F"]["upper"]

_b = plt.get_cmap(BETA_CMAP)
CMAP = LinearSegmentedColormap.from_list("beta_vis", _b(np.linspace(0.28, 1.0, 256)))

T_R_YLIM = (119.0, 142.0)


def main():
    t0 = time.time()
    apply_style()
    d = json.load(open(TRACES))
    C = d["cases"]
    campaign_h = d["campaign_h"]
    beta_final = d["beta_final"]
    tau = campaign_h / np.log(1.0 / beta_final)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharex=True, sharey=True)

    stats = {}
    for ax, (case, title) in zip(axes, PANELS):
        dec = C[case]["decisions"]
        t = np.array([x["t"] for x in dec])
        Fr = np.array([x["F_ref"] for x in dec])
        Tr = np.array([x["T_R_ref"] for x in dec])
        Fs = np.array([x["F_s"] for x in dec])
        Ts = np.array([x["T_R_s"] for x in dec])
        beta = np.exp(-t / tau)

        disp = np.hypot((Fr - Fs) / (F_HI - F_LO), (Tr - Ts) / 50.0)

        ax.quiver(Fr, Tr, Fs - Fr, Ts - Tr, beta, cmap=CMAP,
                  angles="xy", scale_units="xy", scale=1.0, width=0.004,
                  headwidth=4, headlength=5, alpha=0.9, zorder=3)
        sc = ax.scatter(Fs, Ts, c=beta, cmap=CMAP, s=9, zorder=4,
                        edgecolors="none")

        ax.axhline(T_LIMIT, color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.2,
                   zorder=2)
        ax.axvline(F_HI, color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.2,
                   zorder=2)
        ax.set_title(f"{title}\nmedian projection displacement "
                     f"{np.median(disp):.3f} of range", fontsize=9)
        ax.set_xlabel("$F$ (h$^{-1}$)")
        ax.set_ylim(*T_R_YLIM)

        stats[case] = {"median_displacement_frac_of_range": float(np.median(disp)),
                       "mean_displacement_frac_of_range": float(disp.mean()),
                       "p90_displacement": float(np.percentile(disp, 90)),
                       "frac_requests_moved_over_5pct": float((disp > 0.05).mean())}

    axes[0].set_ylabel("$T_R$ ($\\degree$C)")
    axes[0].annotate(f"temperature limit", xy=(0.02, T_LIMIT),
                     xycoords=("axes fraction", "data"), xytext=(0, 3),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=7.5, color=CONSTRAINT_COLOR)
    axes[1].annotate("throughput bound", xy=(F_HI, 0.02),
                     xycoords=("data", "axes fraction"), xytext=(-4, 0),
                     textcoords="offset points", ha="right", va="bottom",
                     rotation=90, fontsize=7.5, color=CONSTRAINT_COLOR)

    cb = fig.colorbar(sc, ax=axes, pad=0.015, fraction=0.04)
    cb.set_label("catalyst activity $\\beta$", fontsize=8)

    fig.suptitle("Each arrow runs from the supervisor's request to the target "
                 "the plant was actually given", fontsize=10)
    fig.text(0.5, -0.06,
             "One arrow per supervisory decision over a severe campaign. "
             "Displacement is normalised by each axis's operating range so the "
             "two coordinates are\ncomparable. The learned tier's requests are "
             "moved an order of magnitude further than the classical layer's, "
             "which is why an\nimprovement in request accuracy need not reach "
             "profit. The temperature axis is clipped at 119 $\\degree$C; one "
             "learned request falls below it.",
             ha="center", va="top", fontsize=7.5, color=TEXT_SECONDARY)

    path = "supervisors/request_vs_realised.png"
    save(fig, path,
         params={"traces": TRACES, "cases": [c for c, _ in PANELS],
                 "T_limit": T_LIMIT, "F_bounds": [F_LO, F_HI],
                 "tau_h": float(tau), "T_R_ylim": list(T_R_YLIM)},
         script=__file__, start_time=t0, extra=stats)

    print(f"{'case':14} {'median disp':>12} {'mean':>8} {'p90':>8} {'>5% of range':>13}")
    for k, v in stats.items():
        print(f"{k:14} {v['median_displacement_frac_of_range']:12.4f} "
              f"{v['mean_displacement_frac_of_range']:8.4f} "
              f"{v['p90_displacement']:8.4f} "
              f"{100*v['frac_requests_moved_over_5pct']:12.1f}%")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
