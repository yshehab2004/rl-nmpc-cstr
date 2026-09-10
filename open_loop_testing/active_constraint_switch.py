import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state

Q_DOT_FLOOR = -8500.0


def cooling_floor_T_R(F, beta, alpha, C_A0, T_in, C,
                        T_R_lo=100.0, T_R_hi=150.0, tol=1e-9, maxit=200):
    def slack(T_R):
        _, _, _, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
        return float(Q_dot) - Q_DOT_FLOOR

    lo, hi = T_R_lo, T_R_hi
    slack_lo, slack_hi = slack(lo), slack(hi)
    if slack_lo >= 0:
        return lo
    if slack_hi < 0:
        return np.inf

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        slack_mid = slack(mid)
        if abs(slack_mid) < tol or (hi - lo) < 1e-12:
            return mid
        if slack_mid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def peak_cooling_floor_T_R(beta, alpha, C_A0, T_in, C, F_range=(5.0, 35.0), n_F=121,
                             T_R_lo=100.0, T_R_hi=150.0):
    F_values = np.linspace(F_range[0], F_range[1], n_F)
    boundaries = np.array([
        cooling_floor_T_R(F, beta, alpha, C_A0, T_in, C, T_R_lo, T_R_hi)
        for F in F_values
    ])
    i_max = int(np.argmax(boundaries))
    return float(boundaries[i_max]), float(F_values[i_max])


def find_switch_beta(beta_lo, beta_hi, alpha, C_A0, T_in, C, T_R_max=140.0,
                       F_range=(5.0, 35.0), n_F=121, tol=1e-6, maxit=100):
    def gap(beta):
        peak, _ = peak_cooling_floor_T_R(beta, alpha, C_A0, T_in, C, F_range, n_F)
        return peak - T_R_max

    lo, hi = beta_lo, beta_hi
    gap_lo, gap_hi = gap(lo), gap(hi)
    if gap_lo > 0 and gap_hi > 0:
        raise ValueError(f"no switch in [{beta_lo}, {beta_hi}]: temperature limit "
                          f"already binding at both ends (peaks {gap_lo + T_R_max:.2f}, "
                          f"{gap_hi + T_R_max:.2f})")
    if gap_lo < 0 and gap_hi < 0:
        raise ValueError(f"no switch in [{beta_lo}, {beta_hi}]: cooling floor "
                          f"binding throughout (peaks {gap_lo + T_R_max:.2f}, "
                          f"{gap_hi + T_R_max:.2f})")

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        gap_mid = gap(mid)
        if abs(gap_mid) < tol:
            return mid
        if (gap_mid < 0) == (gap_hi < 0):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    import time

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"],
              C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    beta_sweep = [0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
    print("Peak cooling-floor T_R (worst case, over F in [5,35]) by beta:")
    peak_table = {}
    for beta in beta_sweep:
        peak, F_at_peak = peak_cooling_floor_T_R(beta, **kw)
        binds = peak >= 140.0
        peak_table[f"{beta:.2f}"] = {"peak_required_T_R": peak, "F_at_peak": F_at_peak,
                                      "temp_limit_binds": binds}
        print(f"  beta = {beta:.2f}: peak required T_R = {peak:7.3f} degC "
              f"at F = {F_at_peak:5.2f}  [{'TEMP LIMIT BINDS' if binds else 'cooling floor only'}]")

    beta_star = find_switch_beta(beta_lo=0.35, beta_hi=0.40, T_R_max=140.0, **kw)
    print(f"\nSwitch beta* = {beta_star:.5f} "
          f"(temperature limit first becomes binding here, at F = 35 h^-1)")

    out_path = "open_loop_testing/active_constraint_switch.json"
    with open(out_path, "w") as f:
        import json
        json.dump({"peak_cooling_floor_T_R_by_beta": peak_table, "switch_beta": beta_star}, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"beta_sweep": beta_sweep, "beta_lo": 0.35, "beta_hi": 0.40, "T_R_max": 140.0,
                "F_range": [5.0, 35.0], "n_F": 121, **{k: v for k, v in kw.items() if k != "C"}},
        script=__file__, start_time=start_time,
        extra={"switch_beta": beta_star},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")
