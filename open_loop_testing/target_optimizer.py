import numpy as np

from open_loop_testing.active_constraint_multipliers import _T_R_for_cooling_floor
from open_loop_testing.steady_state_map import solve_steady_state

F_BOUNDS_DEFAULT = (5.0, 35.0)
T_R_BOUNDS_DEFAULT = (100.0, 140.0)


def _is_feasible(F, T_R, margin, beta_hat, C_A0, alpha, T_in, C,
                  q_lower, q_upper, T_R_max, tol=0.0):
    _, _, _, Q_dot = solve_steady_state(F, T_R, beta_hat, alpha, C_A0, T_in, C)
    Q_dot = float(Q_dot)
    if T_R > T_R_max - margin + tol:
        return False
    if Q_dot > q_upper + tol:
        return False
    floor = _T_R_for_cooling_floor(F, beta_hat, alpha, C_A0, T_in, C, q_lower=q_lower)
    return T_R >= floor + margin - tol


def _t_r_at_q_dot(F_grid, q_target, beta_hat, alpha, C_A0, T_in, C,
                    T_R_lo=100.0, T_R_hi=150.0, iters=60):
    lo = np.full(np.shape(F_grid), T_R_lo, dtype=float)
    hi = np.full(np.shape(F_grid), T_R_hi, dtype=float)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, _, _, Q_dot = solve_steady_state(F_grid, mid, beta_hat, alpha, C_A0, T_in, C)
        below = np.asarray(Q_dot) < q_target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _grid_projection(F_ref, T_R_ref, margin, beta_hat, C_A0, alpha, T_in, C,
                      q_lower, q_upper, T_R_max, F_bounds, T_R_bounds, n_F=1201):
    F_grid = np.linspace(F_bounds[0], F_bounds[1], n_F)
    kw = dict(beta_hat=beta_hat, alpha=alpha, C_A0=C_A0, T_in=T_in, C=C)
    lo = _t_r_at_q_dot(F_grid, q_lower, **kw) + margin
    hi = np.minimum(T_R_max - margin, _t_r_at_q_dot(F_grid, q_upper, **kw))

    feasible = lo <= hi
    if not np.any(feasible):
        raise ValueError(
            f"no feasible (F, T_R) at beta_hat={beta_hat:.4f}, C_A0={C_A0:.3f}, "
            f"margin={margin}: the plant has no operating point under these "
            f"beliefs. That is a physical result, not a solver failure.")

    F_range = F_bounds[1] - F_bounds[0]
    T_R_range = T_R_bounds[1] - T_R_bounds[0]

    def best_on(F_axis):
        f_lo = _t_r_at_q_dot(F_axis, q_lower, **kw) + margin
        f_hi = np.minimum(T_R_max - margin, _t_r_at_q_dot(F_axis, q_upper, **kw))
        ok = f_lo <= f_hi
        T_best = np.clip(T_R_ref, f_lo, f_hi)
        obj = (((F_axis - F_ref) / F_range) ** 2
               + ((T_best - T_R_ref) / T_R_range) ** 2)
        return np.where(ok, obj, np.inf), T_best, ok

    obj, T_R_best, _ = best_on(F_grid)
    i = int(np.argmin(obj))

    step = F_grid[1] - F_grid[0]
    fine = np.linspace(max(F_bounds[0], F_grid[i] - step),
                        min(F_bounds[1], F_grid[i] + step), 201)
    obj_f, T_R_fine, ok_f = best_on(fine)
    if np.any(np.isfinite(obj_f)):
        j = int(np.argmin(obj_f))
        return float(fine[j]), float(T_R_fine[j])
    return float(F_grid[i]), float(T_R_best[i])


def solve_target(F_ref, T_R_ref, margin, beta_hat, C_A0, alpha, T_in, C,
                  q_lower=-8500.0, q_upper=0.0, T_R_max=140.0,
                  F_bounds=F_BOUNDS_DEFAULT, T_R_bounds=T_R_BOUNDS_DEFAULT):
    T_R_max_eff = T_R_max - margin

    def T_R_floor(F):
        return _T_R_for_cooling_floor(F, beta_hat, alpha, C_A0, T_in, C, q_lower=q_lower) + margin

    if _is_feasible(F_ref, T_R_ref, margin, beta_hat, C_A0, alpha, T_in, C,
                     q_lower, q_upper, T_R_max):
        return {"F_s": F_ref, "T_R_s": T_R_ref, "projected": False}

    F_s, T_R_s = _grid_projection(
        F_ref, T_R_ref, margin, beta_hat, C_A0, alpha, T_in, C,
        q_lower, q_upper, T_R_max, F_bounds, T_R_bounds)
    return {"F_s": F_s, "T_R_s": T_R_s, "projected": True}
