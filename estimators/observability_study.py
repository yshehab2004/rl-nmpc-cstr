import json
import time

import numpy as np

from estimators.ekf import H, N_AUG, N_PHYS, BetaEKF
from models.params import load_params
from open_loop_testing.optimum_trajectory import find_optimum
from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.linearization import ode_rhs
from utils.provenance import save_with_provenance

P = load_params()
C = P["certain_params"]
FEED = P["nominal_feed"]
ECON = P["economics"]
ALPHA = P["uncertain_params"]["alpha_nominal"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])

BETAS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)

F_GRID = np.linspace(5.0, 35.0, 601)
T_R_GRID = np.linspace(100.0, 150.0, 501)

SCALE = np.array([5.1, 1.25, 50.0, 50.0, 1.0])
C_A0_SCALE = 1.2


def jacobian_beta(x_aug, u, C_A0, T_in, eps=1e-6):
    def f(xa):
        dx = ode_rhs(xa[:N_PHYS], u, alpha=ALPHA, beta=xa[4],
                     C_A0=C_A0, T_in=T_in, C=C)
        return np.concatenate([dx, [0.0]])
    A = np.zeros((N_AUG, N_AUG))
    for i in range(N_AUG):
        h = eps * max(abs(x_aug[i]), 1.0)
        xp, xm = x_aug.copy(), x_aug.copy()
        xp[i] += h
        xm[i] -= h
        A[:, i] = (f(xp) - f(xm)) / (2 * h)
    return A


def jacobian_beta_and_feed(x_ext, u, T_in, eps=1e-6):
    n = 6
    def f(xe):
        dx = ode_rhs(xe[:N_PHYS], u, alpha=ALPHA, beta=xe[4],
                     C_A0=xe[5], T_in=T_in, C=C)
        return np.concatenate([dx, [0.0, 0.0]])
    A = np.zeros((n, n))
    for i in range(n):
        h = eps * max(abs(x_ext[i]), 1.0)
        xp, xm = x_ext.copy(), x_ext.copy()
        xp[i] += h
        xm[i] -= h
        A[:, i] = (f(xp) - f(xm)) / (2 * h)
    return A


def observability(A, Hm, scale):
    n = A.shape[0]
    blocks, M = [], np.eye(n)
    for _ in range(n):
        blocks.append(Hm @ M)
        M = M @ A
    O = np.vstack(blocks)
    O_scaled = O * scale[None, :]
    s = np.linalg.svd(O_scaled, compute_uv=False)
    tol = max(O_scaled.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tol))
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return {"rank": rank, "n_states": n, "singular_values": s.tolist(),
            "condition": cond, "smallest_sv": float(s[-1])}


def main():
    t_start = time.time()
    H6 = np.zeros((2, 6))
    H6[0, 2] = 1.0
    H6[1, 1] = 1.0
    scale5 = SCALE
    scale6 = np.concatenate([SCALE, [C_A0_SCALE]])

    rows = []
    for beta in BETAS:
        opt = find_optimum(F_GRID, T_R_GRID, beta, ALPHA, FEED["C_A0"],
                           FEED["T_in"], C, **PRICES, T_R_max=140.0)
        F_s, T_R_s = opt["F_star"], opt["T_R_star"]
        C_a, C_b, T_K, Q_dot = solve_steady_state(F_s, T_R_s, beta, ALPHA,
                                                  FEED["C_A0"], FEED["T_in"], C)
        x = np.array([float(C_a), float(C_b), float(T_R_s), float(T_K)])
        u = np.array([F_s, float(Q_dot)])

        A5 = jacobian_beta(np.concatenate([x, [beta]]), u,
                           FEED["C_A0"], FEED["T_in"])
        o5 = observability(A5, H, scale5)

        A6 = jacobian_beta_and_feed(
            np.concatenate([x, [beta, FEED["C_A0"]]]), u, FEED["T_in"])
        o6 = observability(A6, H6, scale6)

        rows.append({"beta": beta, "F_star": F_s, "T_R_star": T_R_s,
                     "beta_only": o5, "beta_and_C_A0": o6})

    print("Observability along the optimum trajectory, from y = (T_R, C_b).\n")
    hdr = (f"{'beta':>5} {'F*':>7} {'T_R*':>7} | {'rank':>5} {'cond':>11} "
           f"| {'rank':>5} {'cond':>11}")
    print(f"{'':>21} |  (x, beta)        |  (x, beta, C_A0)")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        a, b = r["beta_only"], r["beta_and_C_A0"]
        print(f"{r['beta']:5.2f} {r['F_star']:7.2f} {r['T_R_star']:7.1f} | "
              f"{a['rank']:2d}/{a['n_states']:<2d} {a['condition']:11.3e} | "
              f"{b['rank']:2d}/{b['n_states']:<2d} {b['condition']:11.3e}")

    full5 = all(r["beta_only"]["rank"] == N_AUG for r in rows)
    full6 = all(r["beta_and_C_A0"]["rank"] == 6 for r in rows)
    worst6 = max(r["beta_and_C_A0"]["condition"] for r in rows)
    worst5 = max(r["beta_only"]["condition"] for r in rows)
    ratio = worst6 / worst5 if worst5 else float("inf")

    print()
    print(f"(x, beta)        full rank at every beta: {full5}")
    print(f"(x, beta, C_A0)  full rank at every beta: {full6}")
    print(f"worst condition, beta only     : {worst5:.3e}")
    print(f"worst condition, beta and C_A0 : {worst6:.3e}")
    print(f"conditioning penalty for estimating C_A0: {ratio:.1f}x")

    out = {"rows": rows,
           "beta_only_full_rank_everywhere": full5,
           "beta_and_C_A0_full_rank_everywhere": full6,
           "worst_condition_beta_only": worst5,
           "worst_condition_beta_and_C_A0": worst6,
           "conditioning_penalty": ratio,
           "measurements": ["T_R", "C_b"],
           "scaling": {"C_a": 5.1, "C_b": 1.25, "T_R": 50.0, "T_K": 50.0,
                       "beta": 1.0, "C_A0": C_A0_SCALE}}
    path = "estimators/observability_study.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    save_with_provenance(
        path, None,
        params={"betas": list(BETAS), "T_R_max": 140.0,
                "F_grid": [5.0, 35.0, len(F_GRID)],
                "T_R_grid": [100.0, 150.0, len(T_R_GRID)],
                "jacobian_eps": 1e-6},
        script=__file__, start_time=t_start,
        extra={"conditioning_penalty": ratio,
               "beta_only_full_rank_everywhere": full5,
               "beta_and_C_A0_full_rank_everywhere": full6})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
