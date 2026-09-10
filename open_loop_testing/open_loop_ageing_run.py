import numpy as np
from scipy.integrate import solve_ivp

from open_loop_testing.linearization import ode_rhs


def simulate_open_loop_ageing(x0, u_fixed, schedule, campaign_h, alpha, C_A0, T_in, C, n=300):
    def rhs(t, x):
        _, beta = schedule(t)
        return ode_rhs(x, u_fixed, alpha, beta, C_A0, T_in, C)

    t_eval = np.linspace(0.0, campaign_h, n)
    sol = solve_ivp(rhs, (0.0, campaign_h), x0, t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-8)
    betas = np.array([schedule(t)[1] for t in sol.t])
    return {"t": sol.t, "x": sol.y.T, "beta": betas}


if __name__ == "__main__":
    import json
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models.params import load_params
    from models.deactivation import make_severity_fn
    from open_loop_testing.steady_state_map import solve_steady_state
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    CAMPAIGN_H = 75.0
    schedule = make_severity_fn("severe", CAMPAIGN_H, profile="exponential")

    policies = {
        "beta=1.0 optimum (F=35, sized for fresh catalyst)": (35.0, 133.0),
        "beta=0.6 optimum (F=35, T_R=140)": (35.0, 140.0),
        "T8 best always-feasible (F=18.25, T_R=140)": (18.25, 140.0),
    }

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    results = {}
    for label, (F0, T_R0) in policies.items():
        C_a0, C_b0, T_K0, Q_dot0 = solve_steady_state(F0, T_R0, 1.0, **kw)
        x0 = np.array([C_a0, C_b0, T_R0, T_K0])
        u_fixed = np.array([F0, Q_dot0])
        sim = simulate_open_loop_ageing(x0, u_fixed, schedule, CAMPAIGN_H, **kw)
        axes[0].plot(sim["t"], sim["x"][:, 2], label=label)
        axes[1].plot(sim["t"], sim["beta"], label=label)
        results[label] = {"F": F0, "Q_dot0": float(Q_dot0),
                            "T_R_final": float(sim["x"][-1, 2]), "T_R_initial": T_R0,
                            "T_R_max": float(sim["x"][:, 2].max())}
        print(f"{label}: T_R {T_R0:.2f} -> {sim['x'][-1,2]:.2f} degC over {CAMPAIGN_H}h "
              f"(peak {sim['x'][:,2].max():.2f})")

    axes[0].axhline(140.0, color="red", ls=":", lw=1, label="140 degC limit")
    axes[0].set_ylabel("T_R (degC)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Open-loop ageing run: T_R drift with F, Q_dot fixed, beta decaying on the real schedule")
    axes[1].set_xlabel("t (h)")
    axes[1].set_ylabel("beta")
    fig.tight_layout()

    out_path = "open_loop_testing/open_loop_ageing_run.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

    json_path = "open_loop_testing/open_loop_ageing_run.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, out_path], data=None,
        params={"campaign_h": CAMPAIGN_H, "profile": "exponential", "severity": "severe",
                "policies": {k: list(v) for k, v in policies.items()}},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {json_path}, {sidecar}")
