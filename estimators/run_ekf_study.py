import json
import time

import numpy as np

from estimators.run_estimator import run
from models.deactivation import make_deactivation_fn
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from open_loop_testing.steady_state_map import solve_steady_state
from utils import plotstyle as ps

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]

F0, T_R0 = 18.83, 134.14
_C_a, _C_b, _T_K, _Q_dot = solve_steady_state(F0, T_R0, 1.0, 1.0,
                                                FEED["C_A0"], FEED["T_in"], C)
X_SS = np.array([float(_C_a), float(_C_b), T_R0, float(_T_K)])
U_SS = np.array([F0, float(_Q_dot)])


def tracking_study():
    out = {}
    for label, campaign_h, n_steps in (("real_75h", 75.0, 15000),
                                        ("compressed_2h", 2.0, 400)):
        ageing_fn = make_deactivation_fn(
            beta_final=0.2 if campaign_h > 10 else 0.5, campaign_h=campaign_h)
        r = run(X_SS, U_SS, n_steps=n_steps, noise=True, seed=1,
                 beta0=1.0, ageing_fn=ageing_fn)
        lag = np.abs(r["beta_true"] - r["beta_hat"])
        out[label] = {"campaign_h": campaign_h, "t": r["t"].tolist(),
                       "beta_true": r["beta_true"].tolist(),
                       "beta_hat": r["beta_hat"].tolist(),
                       "max_lag": float(lag.max()), "mean_lag": float(lag.mean())}
    return out


def aliasing_study():
    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(FEED["C_A0"], 0.6, 0.25))
    out = {}
    for label, known in (("measured_C_A0", True), ("blind_nominal_C_A0", False)):
        r = run(X_SS, U_SS, n_steps=300, noise=False, beta0=1.0,
                 disturbance_fn=disturbed, C_A0_known=known)
        out[label] = {"t": r["t"].tolist(), "beta_hat": r["beta_hat"].tolist(),
                       "C_A0_true": r["C_A0_true"].tolist(),
                       "beta_hat_final": float(r["beta_hat"][-1]),
                       "phantom_ageing": float(abs(r["beta_hat"][-1] - 1.0))}
    return out


def plot_tracking(data, path, script, start_time=None):
    import matplotlib.pyplot as plt
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, key, title in ((axes[0], "real_75h", "real severe campaign (75 h)"),
                            (axes[1], "compressed_2h", "compressed campaign (2 h)")):
        d = data[key]
        ax.plot(d["t"], d["beta_true"], color=ps.TRUTH_COLOR, lw=1.4, label=r"$\beta$ true")
        ax.plot(d["t"], d["beta_hat"], color=ps.tier("T2_rto")["color"], ls="--",
                label=r"$\hat{\beta}$ estimate")
        ax.set_xlabel("time (h)")
        ax.set_title(f"{title}\nmax lag {d['max_lag']:.3f}", fontsize=9)
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel(r"catalyst activity $\beta$")
    axes[0].legend(loc="lower left")
    fig.suptitle("EKF tracking: an accelerated campaign is not a conservative proxy",
                  fontsize=10)
    return ps.save(fig, path, {"study": "tracking"}, script, start_time=start_time)


def plot_aliasing(data, path, script, start_time=None):
    import matplotlib.pyplot as plt
    ps.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)

    d0 = data["measured_C_A0"]
    axes[0].plot(d0["t"], d0["C_A0_true"], color=ps.REFERENCE_COLOR, lw=1.4)
    axes[0].set_ylabel("C$_{A0}$ (mol L$^{-1}$)")
    axes[0].set_title("feed concentration steps 5.1 $\\rightarrow$ 5.7 at 0.25 h",
                       fontsize=8, color=ps.TEXT_SECONDARY, loc="right")

    ax = axes[1]
    ax.axhline(1.0, color=ps.TRUTH_COLOR, lw=1.4, label=r"$\beta$ true (constant)")
    for key, style, lbl in (("measured_C_A0", "-", "estimator given measured C$_{A0}$"),
                             ("blind_nominal_C_A0", "--", "estimator given nominal C$_{A0}$")):
        d = data[key]
        col = ps.tier("T2_rto")["color"] if key == "measured_C_A0" else ps.CONSTRAINT_COLOR
        ax.plot(d["t"], d["beta_hat"], color=col, ls=style, label=lbl)
    blind = data["blind_nominal_C_A0"]["beta_hat_final"]
    ax.annotate(f"phantom ageing {100*(1-blind):.0f}%",
                 xy=(d["t"][-1], blind), xytext=(-120, 12),
                 textcoords="offset points", color=ps.CONSTRAINT_COLOR, fontsize=8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel(r"$\hat{\beta}$")
    ax.set_ylim(0.6, 1.08)
    ax.legend(loc="lower left")
    fig.suptitle("Withholding a measurable input manufactures catalyst deactivation",
                  fontsize=10)
    return ps.save(fig, path, {"study": "aliasing"}, script, start_time=start_time)


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    tracking = tracking_study()
    aliasing = aliasing_study()

    print("F-038 tracking:")
    for k, d in tracking.items():
        print(f"  {k:14} max lag {d['max_lag']:.4f}  mean {d['mean_lag']:.4f}")
    print("F-039 aliasing (beta_true constant at 1.0):")
    for k, d in aliasing.items():
        print(f"  {k:20} beta_hat final {d['beta_hat_final']:.4f}  "
              f"phantom {d['phantom_ageing']:.4f}")

    plot_tracking(tracking, "estimators/ekf_tracking.png", __file__)
    plot_aliasing(aliasing, "estimators/ekf_aliasing.png", __file__)

    out_path = "estimators/ekf_study.json"
    with open(out_path, "w") as f:
        json.dump({"tracking": tracking, "aliasing": aliasing}, f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"noise_frac": PARAMS["measurement"]["noise_frac"],
                "estimator": PARAMS["estimator"], "x0": X_SS.tolist(),
                "u": U_SS.tolist(), "C_A0_step": 0.6, "seed": 1},
        script=__file__, start_time=start_time,
        extra={"tracking_summary": {k: {"max_lag": v["max_lag"], "mean_lag": v["mean_lag"]}
                                     for k, v in tracking.items()},
               "aliasing_summary": {k: v["phantom_ageing"] for k, v in aliasing.items()}})
    print(f"Saved {out_path}, {sidecar}")
