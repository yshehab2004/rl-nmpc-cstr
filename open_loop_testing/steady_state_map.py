import numpy as np


def rate_constants(T_R, alpha, beta, C):
    T = np.asarray(T_R, dtype=float) + 273.15
    K1 = beta * C["K0_ab"] * np.exp(-C["E_A_ab"] / T)
    K2 = beta * C["K0_bc"] * np.exp(-C["E_A_bc"] / T)
    K3 = C["K0_ad"] * np.exp(-alpha * C["E_A_ad"] / T)
    return K1, K2, K3


def solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C):
    F = np.asarray(F, dtype=float)
    T_R = np.asarray(T_R, dtype=float)
    K1, K2, K3 = rate_constants(T_R, alpha, beta, C)

    a_ = K3
    b_ = F + K1
    c_ = -F * C_A0
    disc = b_ ** 2 - 4 * a_ * c_
    q = -0.5 * (b_ + np.sqrt(disc))
    C_a = c_ / q

    C_b = K1 * C_a / (F + K2)

    HG = K1 * C_a * C["H_R_ab"] + K2 * C_b * C["H_R_bc"] + K3 * C_a ** 2 * C["H_R_ad"]
    T_K = (
        T_R
        + HG * C["V_R"] / (C["K_w"] * C["A_R"])
        - F * (T_in - T_R) * C["rho"] * C["Cp"] * C["V_R"] / (C["K_w"] * C["A_R"])
    )

    Q_dot = C["K_w"] * C["A_R"] * (T_K - T_R)

    return C_a, C_b, T_K, Q_dot


def build_map(F_grid, T_R_grid, betas, alpha, C_A0, T_in, C, out_path):
    F_grid = np.asarray(F_grid, dtype=float)
    T_R_grid = np.asarray(T_R_grid, dtype=float)
    betas = np.asarray(betas, dtype=float)

    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")
    n_beta, n_F, n_TR = len(betas), len(F_grid), len(T_R_grid)

    C_a = np.empty((n_beta, n_F, n_TR))
    C_b = np.empty((n_beta, n_F, n_TR))
    T_K = np.empty((n_beta, n_F, n_TR))
    Q_dot = np.empty((n_beta, n_F, n_TR))

    for i, beta in enumerate(betas):
        C_a[i], C_b[i], T_K[i], Q_dot[i] = solve_steady_state(
            F_mesh, T_R_mesh, beta, alpha, C_A0, T_in, C
        )

    np.savez(out_path, F=F_grid, T_R=T_R_grid, beta=betas,
              C_a=C_a, C_b=C_b, T_K=T_K, Q_dot=Q_dot)
    return out_path


if __name__ == "__main__":
    import time

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    CERTAIN = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]

    F_range = (5.0, 35.0, 0.25)
    T_R_range = (100.0, 150.0, 0.25)
    F_grid = np.linspace(F_range[0], F_range[1], 121)
    T_R_grid = np.linspace(T_R_range[0], T_R_range[1], 201)
    betas = [1.0, 0.8, 0.6, 0.4, 0.2]

    out_path = build_map(
        F_grid, T_R_grid, np.array(betas),
        alpha=PARAMS["uncertain_params"]["alpha_nominal"],
        C_A0=feed["C_A0"], T_in=feed["T_in"],
        C=CERTAIN, out_path="open_loop_testing/steady_state_map.npz",
    )
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"F_range": list(F_range), "T_R_range": list(T_R_range), "beta": betas,
                "alpha": PARAMS["uncertain_params"]["alpha_nominal"],
                "C_A0": feed["C_A0"], "T_in": feed["T_in"]},
        script=__file__, start_time=start_time,
    )
    print(f"Saved {out_path} "
          f"({len(betas)} beta x {len(F_grid)} F x {len(T_R_grid)} T_R nodes)")
    print(f"Saved {sidecar}")
