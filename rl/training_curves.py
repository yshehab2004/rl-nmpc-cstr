import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotstyle import (SURFACE, TEXT_SECONDARY, TIER_STYLE,
                             apply_style, save)

ARMS = [("A1_new_withmargin", "A1: corrected coverage (uniform in $\\beta$)"),
        ("A3_old_withmargin", "A3: legacy coverage")]

TAIL = 20
SMOOTH = 15
WARMUP = 5


def load_arm(arm):
    out = {}
    for d in sorted(glob.glob(f"rl/runs_d028/{arm}/*/")):
        path = os.path.join(d, "monitor.monitor.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, skiprows=1)
        seed = int(d.rstrip("/").split("seed")[1].split("_")[0])
        out[seed] = df["r"].to_numpy()
    return dict(sorted(out.items()))


def main():
    t0 = time.time()
    apply_style()

    data = {arm: load_arm(arm) for arm, _ in ARMS}
    stats = {}
    for arm, _ in ARMS:
        finals = np.array([r[-TAIL:].mean() for r in data[arm].values()])
        stats[arm] = {"min": float(finals.min()), "max": float(finals.max()),
                      "median": float(np.median(finals)),
                      "ratio": float(finals.max() / finals.min()),
                      "per_seed": {int(s): float(r[-TAIL:].mean())
                                   for s, r in data[arm].items()}}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    rl_colour = TIER_STYLE["T3_rl"]["color"]

    smoothed = np.concatenate([
        pd.Series(r).rolling(SMOOTH, min_periods=1).mean().to_numpy()[WARMUP:]
        for a, _ in ARMS for r in data[a].values()])
    lo, hi = smoothed.min(), smoothed.max()
    pad = 0.10 * (hi - lo)

    for ax, (arm, title) in zip(axes, ARMS):
        seeds = data[arm]
        st = stats[arm]
        worst = min(st["per_seed"], key=st["per_seed"].get)
        best = max(st["per_seed"], key=st["per_seed"].get)

        for seed, r in seeds.items():
            s = pd.Series(r).rolling(SMOOTH, min_periods=1).mean()
            emphasis = seed in (best, worst)
            ax.plot(np.arange(len(s)), s, color=rl_colour,
                    lw=1.7 if emphasis else 0.8,
                    alpha=1.0 if emphasis else 0.28,
                    zorder=3 if emphasis else 2)

        for seed, style, frac in ((best, "best", 0.93), (worst, "worst", 0.07)):
            r = pd.Series(seeds[seed]).rolling(SMOOTH, min_periods=1).mean()
            ax.annotate(f"seed {seed}, {style}: {st['per_seed'][seed]:.0f}",
                        xy=(len(r) - 1, r.iloc[-1]), xycoords="data",
                        xytext=(0.06, frac), textcoords="axes fraction",
                        ha="left", va="center", fontsize=7.5,
                        color=TEXT_SECONDARY,
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE,
                                  ec="#e6e5e1", lw=0.6),
                        arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY,
                                        lw=0.6, alpha=0.55,
                                        shrinkA=2, shrinkB=2))

        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{title}\nfinal return spans "
                     f"{st['ratio']:.2f}$\\times$ across seeds", fontsize=9)
        ax.set_xlabel("training episode")

    axes[0].set_ylabel("episode return (reward scale)")

    fig.suptitle("Ten seeds per arm, identical settings, differing only in "
                 "random seed", fontsize=10)
    fig.text(0.5, -0.06,
             "Panels are not comparable in level: uniform-in-$\\beta$ sampling "
             "draws harder episodes, lowering return independently of policy "
             "quality.\nOnly the spread within a panel is meaningful. The "
             "opening episodes of an untrained policy fall below the axis.",
             ha="center", va="top", fontsize=7.5, color=TEXT_SECONDARY)

    path = "rl/training_curves.png"
    save(fig, path,
         params={"arms": [a for a, _ in ARMS], "tail_episodes": TAIL,
                 "smoothing_window": SMOOTH, "warmup_excluded": WARMUP,
                 "seeds_per_arm": 10},
         script=__file__, start_time=t0, extra=stats)

    for arm, _ in ARMS:
        s = stats[arm]
        print(f"{arm:22} min={s['min']:7.1f} max={s['max']:7.1f} "
              f"median={s['median']:7.1f} ratio={s['ratio']:.2f}x")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
