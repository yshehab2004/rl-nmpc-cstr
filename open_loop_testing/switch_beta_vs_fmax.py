import numpy as np

from open_loop_testing.active_constraint_switch import peak_cooling_floor_T_R, find_switch_beta


def _find_bracket(F_max, alpha, C_A0, T_in, C, T_R_max=140.0, n_scan=40):
    betas = np.linspace(0.05, 1.0, n_scan)
    gaps = []
    for beta in betas:
        peak, _ = peak_cooling_floor_T_R(beta, alpha, C_A0, T_in, C, F_range=(5.0, F_max))
        gaps.append(peak - T_R_max)
    gaps = np.array(gaps)

    sign_changes = np.where(np.diff(np.sign(gaps)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(f"no sign change found scanning beta in [0.05,1.0] for F_max={F_max} "
                          f"(gaps ranged {gaps.min():.2f} to {gaps.max():.2f})")
    i = sign_changes[0]
    return float(betas[i]), float(betas[i + 1])


def switch_beta_for_fmax(F_max, alpha, C_A0, T_in, C, T_R_max=140.0):
    beta_lo, beta_hi = _find_bracket(F_max, alpha, C_A0, T_in, C, T_R_max)
    return find_switch_beta(beta_lo, beta_hi, alpha, C_A0, T_in, C, T_R_max=T_R_max,
                              F_range=(5.0, F_max))


if __name__ == "__main__":
    import json
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

    F_max_values = [25.0, 30.0, 35.0, 40.0, 45.0]
    results = [{"F_max": F_max, "switch_beta": switch_beta_for_fmax(F_max, **kw)} for F_max in F_max_values]

    print(f"{'F_max':>7} {'switch_beta':>12}")
    for r in results:
        print(f"{r['F_max']:7.1f} {r['switch_beta']:12.5f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["F_max"] for r in results], [r["switch_beta"] for r in results], "o-", color="#762a83")
    ax.axvline(35.0, color="gray", ls=":", lw=1, label="current F_max=35")
    ax.set_xlabel("F_max (h$^{-1}$)")
    ax.set_ylabel("switch beta*")
    ax.set_title("Active-constraint switch beta vs. throughput ceiling")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = "open_loop_testing/switch_beta_vs_fmax.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    json_path = "open_loop_testing/switch_beta_vs_fmax.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, out_path], data=None,
        params={"F_max_values": F_max_values, **{k: v for k, v in kw.items() if k != "C"}},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {out_path}, {json_path}, {sidecar}")
