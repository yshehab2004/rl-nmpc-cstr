import numpy as np

from open_loop_testing.linearization import linearize


def steady_state_gain(F, T_R, beta, alpha, C_A0, T_in, C):
    from open_loop_testing.steady_state_map import solve_steady_state
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    A, B = linearize(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), alpha, beta, C_A0, T_in, C)

    G_full = -np.linalg.solve(A, B)
    output_rows = [1, 2]
    return G_full[output_rows, :]


def relative_gain_array(G):
    return G * np.linalg.inv(G).T


if __name__ == "__main__":
    import json
    import time

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    with open("open_loop_testing/optimum_trajectory.json") as f:
        table = json.load(f)

    print(f"{'beta':>6} {'F':>6} {'T_R':>7} {'dCb/dF':>9} {'dCb/dQd':>10} "
          f"{'dTR/dF':>8} {'dTR/dQd':>9} {'RGA[Cb,F]':>10}")
    results = []
    for row in table:
        G = steady_state_gain(row["F_star"], row["T_R_star"], row["beta"], **kw)
        Lambda = relative_gain_array(G)
        results.append({
            "beta": row["beta"], "F": row["F_star"], "T_R": row["T_R_star"],
            "G": G.tolist(), "RGA": Lambda.tolist(),
        })
        print(f"{row['beta']:6.3f} {row['F_star']:6.2f} {row['T_R_star']:7.2f} "
              f"{G[0,0]:9.4f} {G[0,1]:10.6f} {G[1,0]:8.4f} {G[1,1]:9.6f} {Lambda[0,0]:10.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas = [r["beta"] for r in results]
    rga11 = [r["RGA"][0][0] for r in results]
    dTR_dF = [r["G"][1][0] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(betas, rga11, "o-", color="#762a83")
    axes[0].axhline(0.0, color="red", ls=":", lw=1, label="RGA=0 (pairing danger)")
    axes[0].axhline(1.0, color="gray", ls=":", lw=1, label="RGA=1 (ideal decoupling)")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("RGA[C_b, F]")
    axes[0].set_title("Pairing quality (F<->C_b, Q_dot<->T_R)")
    axes[0].legend(fontsize=7)

    axes[1].plot(betas, dTR_dF, "s-", color="#1b7837")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("dT_R/dF at fixed Q_dot (degC per h$^{-1}$)")
    axes[1].set_title("Dilution-cooling magnitude (D4)")
    fig.tight_layout()

    fig_path = "open_loop_testing/rga_and_dilution_cooling.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    out_path = "open_loop_testing/rga.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        [out_path, fig_path], data=None, params={}, script=__file__, start_time=start_time,
        extra={"results": results},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")
