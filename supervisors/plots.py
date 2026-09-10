import numpy as np
import matplotlib.pyplot as plt

from models.params import load_params
from utils import plotstyle as ps

PARAMS = load_params()
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
Q_LOWER = PARAMS["input_bounds"]["Q_dot"]["lower"]
F_BOUNDS = (PARAMS["input_bounds"]["F"]["lower"], PARAMS["input_bounds"]["F"]["upper"])


def plot_campaign(run, tier_name, title, path, script, params=None, start_time=None):
    ps.apply_style()
    st = ps.tier(tier_name)
    t = run["t"]

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.2), sharex=True)

    ax = axes[0]
    ax.plot(t, run["x"][:, 2], color=st["color"], ls="-", label="T$_R$ actual")
    ax.plot(t, run["T_R_s"], color=ps.REFERENCE_COLOR, ls=":", lw=1.1,
            label="T$_R$ commanded")
    ps.limit_line(ax, T_LIMIT, label=f"limit {T_LIMIT:.0f}$\\,^\\circ$C")
    ax.set_ylabel("T$_R$  ($^\\circ$C)")
    ax.legend(loc="best", ncol=3)
    viol = np.maximum(0.0, run["x"][:, 2] - T_LIMIT)
    if viol.max() > 0:
        ax.annotate(f"peak +{viol.max():.2f}$\\,^\\circ$C",
                     xy=(t[int(np.argmax(viol))], run["x"][:, 2].max()),
                     xytext=(4, -10), textcoords="offset points",
                     color=ps.CONSTRAINT_COLOR, fontsize=8)

    ax = axes[1]
    ax.plot(t, run["u"][:, 0], color=st["color"], ls="-", label="F actual")
    ax.plot(t, run["F_s"], color=ps.REFERENCE_COLOR, ls=":", lw=1.1, label="F commanded")
    F = run["u"][:, 0]
    span = max(F.max() - F.min(), 1.0)
    for bound, name in ((F_BOUNDS[1], "F upper bound"), (F_BOUNDS[0], "F lower bound")):
        if min(abs(F.max() - bound), abs(F.min() - bound)) < 0.25 * span:
            ax.axhline(bound, color=ps.CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.0,
                        label=name)
    ax.set_ylabel("F  (h$^{-1}$)")
    ax.legend(loc="best", ncol=3)

    ax = axes[2]
    ax.plot(t, run["beta_true"], color=ps.TRUTH_COLOR, ls="-", lw=1.3, label=r"$\beta$ true")
    ax.plot(t, run["beta_hat"], color=st["color"], ls="--", label=r"$\hat{\beta}$ estimate")
    ax.set_ylabel(r"catalyst activity $\beta$")
    err = np.abs(run["beta_hat"] - run["beta_true"])
    ax.legend(loc="best", ncol=2, title=f"max error {err.max():.3f}")

    ax = axes[3]
    ax.plot(t, run["u"][:, 1], color=st["color"], ls="-", label=r"$\dot{Q}$")
    ps.limit_line(ax, Q_LOWER, label=f"cooling floor {Q_LOWER:.0f}")
    sat = np.mean(run["u"][:, 1] <= Q_LOWER + 1.0)
    ax.set_ylabel(r"$\dot{Q}$  (kJ h$^{-1}$)")
    ax.set_xlabel("time (h)")
    ax.legend(loc="best", ncol=2, title=f"saturated {100*sat:.0f}% of run")

    for a in axes:
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.25 * (hi - lo))
    axes[2].set_ylim(axes[2].get_ylim()[0], 1.02)
    axes[2].legend(loc="upper right", ncol=2,
                    title=f"max error {np.abs(run['beta_hat'] - run['beta_true']).max():.3f}")

    fig.suptitle(title, fontsize=10)
    return ps.save(fig, path, params or {}, script, start_time=start_time)


def plot_tier_comparison(results, path, script, params=None, start_time=None):
    ps.apply_style()
    scenarios = list(results.keys())
    tiers = sorted({t for s in results.values() for t in s["tiers"]})

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)

    panels = [
        (axes[0], "profit_per_hour", "profit (currency h$^{-1}$)", "higher is better"),
        (axes[1], "fraction_time_violating", "fraction of time at/above 140$\\,^\\circ$C",
         "lower is better -- measures headroom, not damage"),
    ]
    for ax, metric, ylabel, note in panels:
        for j, tname in enumerate(tiers):
            st = ps.tier(tname)
            xs, ys = [], []
            for i, scen in enumerate(scenarios):
                m = results[scen]["tiers"].get(tname)
                if m is None:
                    continue
                vals = m[metric] if isinstance(m[metric], list) else [m[metric]]
                offset = (j - (len(tiers) - 1) / 2) * 0.16
                xs.extend([i + offset] * len(vals))
                ys.extend(vals)
                ax.plot([i + offset - 0.05, i + offset + 0.05],
                        [np.mean(vals)] * 2, color=st["color"], lw=2.4, zorder=3)
            ax.scatter(xs, ys, color=st["color"], marker=st["marker"], s=28,
                        zorder=4, label=st["label"], edgecolor=ps.SURFACE, linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(note, fontsize=8, color=ps.TEXT_SECONDARY, loc="right")

    axes[1].axhline(0.0, color=ps.CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.0)
    axes[0].legend(loc="best", ncol=len(tiers))
    axes[1].set_xticks(range(len(scenarios)))
    axes[1].set_xticklabels([s.replace("_", " ") for s in scenarios],
                             rotation=20, ha="right")
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.2 * (hi - lo))
    return ps.save(fig, path, params or {}, script, start_time=start_time)
