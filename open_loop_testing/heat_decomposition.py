import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state


def decompose_Q_dot(F, T_R, beta, alpha, C_A0, T_in, C):
    F = np.asarray(F, dtype=float)
    T_R = np.asarray(T_R, dtype=float)
    beta = np.asarray(beta, dtype=float)

    C_a, C_b, _, _ = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    T = T_R + 273.15
    K1 = beta * C["K0_ab"] * np.exp(-C["E_A_ab"] / T)
    K2 = beta * C["K0_bc"] * np.exp(-C["E_A_bc"] / T)
    K3 = C["K0_ad"] * np.exp(-alpha * C["E_A_ad"] / T)

    Q_feed = F * C["rho"] * C["Cp"] * C["V_R"] * (T_R - T_in)
    Q_AB = C["V_R"] * K1 * C_a * C["H_R_ab"]
    Q_BC = C["V_R"] * K2 * C_b * C["H_R_bc"]
    Q_AD = C["V_R"] * K3 * C_a ** 2 * C["H_R_ad"]
    Q_rxn = Q_AB + Q_BC + Q_AD
    Q_feed = np.broadcast_to(Q_feed, Q_rxn.shape).copy()

    return {"Q_feed": Q_feed, "Q_AB": Q_AB, "Q_BC": Q_BC, "Q_AD": Q_AD, "Q_rxn": Q_rxn}


if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    betas = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    operating_points = [(20.0, 134.0), (35.0, 139.0)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    table = {}
    for ax, (F, T_R) in zip(axes, operating_points):
        terms = decompose_Q_dot(F, T_R, betas, **kw)
        Q_dot_check = terms["Q_feed"] + terms["Q_rxn"]

        key = f"F={F}_T_R={T_R}"
        table[key] = {
            "beta": betas.tolist(),
            "Q_feed": terms["Q_feed"].tolist(),
            "Q_AB": terms["Q_AB"].tolist(), "Q_BC": terms["Q_BC"].tolist(), "Q_AD": terms["Q_AD"].tolist(),
            "Q_rxn": terms["Q_rxn"].tolist(),
            "Q_dot_required": Q_dot_check.tolist(),
        }

        ax.plot(betas, terms["Q_feed"], "o-", color="#1b7837", lw=2, label="Q_feed")
        ax.plot(betas, terms["Q_rxn"], "o-", color="#762a83", lw=2, label="Q_rxn (total)")
        ax.plot(betas, Q_dot_check, "k--", lw=1.2, label="Q_feed + Q_rxn (= Q_dot_required)")
        ax.plot(betas, terms["Q_AD"], ":", color="#762a83", lw=1, alpha=0.6, label="Q_AD (2A->D)")
        ax.axhline(-8500.0, color="red", ls=":", lw=1, label="cooling floor (-8500)")
        ax.axhline(0.0, color="gray", lw=0.6)
        ax.invert_xaxis()
        ax.set_xlabel("beta")
        ax.set_title(f"F = {F} h$^{{-1}}$, T_R = {T_R} degC")
        if ax is axes[0]:
            ax.set_ylabel("kJ/h")

    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Q_dot_required decomposition: feed sensible heat vs. reaction heat")
    fig.tight_layout()

    out_path = "open_loop_testing/heat_decomposition.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    print("Decomposition at each operating point (beta descending 1.0->0.2):")
    for (F, T_R) in operating_points:
        key = f"F={F}_T_R={T_R}"
        print(f"  {key}: Q_feed={table[key]['Q_feed'][0]:.1f} (beta-independent), "
              f"Q_rxn ranges {table[key]['Q_rxn'][0]:.1f} -> {table[key]['Q_rxn'][-1]:.1f}")

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"betas": betas.tolist(), "operating_points": operating_points,
                "alpha": kw["alpha"], "C_A0": kw["C_A0"], "T_in": kw["T_in"]},
        script=__file__, start_time=start_time,
        extra={"decomposition": table},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")
