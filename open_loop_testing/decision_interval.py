import json

import numpy as np

from open_loop_testing.linearization import linearize
from open_loop_testing.nmp_zero import transmission_zeros
from open_loop_testing.steady_state_map import solve_steady_state

BETA_STAR = 0.385
BANDWIDTH_RULE_DIVISOR = 3.0
CANDIDATE_INTERVALS_H = {"15 min": 0.25, "30 min": 0.5, "60 min": 1.0}


def channel_zero(F, T_R, beta, alpha, C_A0, T_in, C, output):
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    A, B = linearize(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), alpha, beta, C_A0, T_in, C)
    if output == "C_b":
        B_col, C_row = B[:, 0], np.array([0.0, 1.0, 0.0, 0.0])
    elif output == "T_R":
        B_col, C_row = B[:, 1], np.array([0.0, 0.0, 1.0, 0.0])
    else:
        raise ValueError(output)
    return transmission_zeros(A, B_col, C_row)


def active_channel(beta):
    return "T_R" if beta > BETA_STAR else "C_b"


def analyze_point(beta, F, T_R, settling_times, alpha, C_A0, T_in, C):
    channel = active_channel(beta)
    zeros = channel_zero(F, T_R, beta, alpha, C_A0, T_in, C, output=channel)
    rhp_zeros = [z.real for z in zeros if z.real > 0]
    min_rhp_zero = min(rhp_zeros) if rhp_zeros else None
    bandwidth_cap = min_rhp_zero / BANDWIDTH_RULE_DIVISOR if min_rhp_zero is not None else None

    st = next(s for s in settling_times if s["beta"] == beta)
    settle_key = "step_Q_settle" if channel == "T_R" else "step_F_settle"
    settling_time_h = st[settle_key][channel]

    ratios = {label: settling_time_h / h for label, h in CANDIDATE_INTERVALS_H.items()}

    return {
        "beta": beta, "F": F, "T_R": T_R,
        "active_channel": f"{'Q_dot' if channel == 'T_R' else 'F'}->{channel}",
        "settling_time_h": settling_time_h,
        "min_rhp_zero": min_rhp_zero,
        "bandwidth_cap_h_inv": bandwidth_cap,
        "ratio_settle_to_interval": ratios,
    }


if __name__ == "__main__":
    import time

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    with open("open_loop_testing/optimum_trajectory.json") as f:
        trajectory = json.load(f)
    with open("open_loop_testing/settling_time.json") as f:
        settling_times = json.load(f)

    results = []
    print(f"{'beta':>6} {'channel':>10} {'settle(h)':>10} {'min RHP z':>10} {'cap(h^-1)':>10} "
          f"{'x15m':>7} {'x30m':>7} {'x60m':>7}")
    for pt in trajectory:
        beta, F, T_R = pt["beta"], pt["F_star"], pt["T_R_star"]
        r = analyze_point(beta, F, T_R, settling_times, **kw)
        results.append(r)
        z_str = f"{r['min_rhp_zero']:.3f}" if r["min_rhp_zero"] is not None else "none"
        cap_str = f"{r['bandwidth_cap_h_inv']:.3f}" if r["bandwidth_cap_h_inv"] is not None else "--"
        ratios = r["ratio_settle_to_interval"]
        print(f"{beta:6.3f} {r['active_channel']:>10} {r['settling_time_h']:10.3f} {z_str:>10} {cap_str:>10} "
              f"{ratios['15 min']:7.2f} {ratios['30 min']:7.2f} {ratios['60 min']:7.2f}")

    out_path = "open_loop_testing/decision_interval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"beta_star": BETA_STAR, "bandwidth_rule_divisor": BANDWIDTH_RULE_DIVISOR,
                "candidate_intervals_h": CANDIDATE_INTERVALS_H},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")
