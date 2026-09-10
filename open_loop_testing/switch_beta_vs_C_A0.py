import numpy as np

from open_loop_testing.active_constraint_switch import peak_cooling_floor_T_R, find_switch_beta


def _find_bracket(C_A0, alpha, T_in, C, T_R_max=140.0, n_scan=60):
    betas = np.linspace(0.05, 1.0, n_scan)
    gaps = []
    for beta in betas:
        peak, _ = peak_cooling_floor_T_R(beta, alpha, C_A0, T_in, C, F_range=(5.0, 35.0))
        gaps.append(peak - T_R_max)
    gaps = np.array(gaps)
    sign_changes = np.where(np.diff(np.sign(gaps)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(f"no sign change scanning beta in [0.05,1.0] for C_A0={C_A0}")
    i = sign_changes[0]
    return float(betas[i]), float(betas[i + 1])


def switch_beta_for_C_A0(C_A0, alpha, T_in, C, T_R_max=140.0):
    beta_lo, beta_hi = _find_bracket(C_A0, alpha, T_in, C, T_R_max)
    return find_switch_beta(beta_lo, beta_hi, alpha, C_A0, T_in, C, T_R_max=T_R_max, F_range=(5.0, 35.0))


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
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], T_in=PARAMS["nominal_feed"]["T_in"], C=C)

    C_A0_values = [4.5, 5.1, 5.7]
    results = [{"C_A0": C_A0, "switch_beta": switch_beta_for_C_A0(C_A0, **kw)} for C_A0 in C_A0_values]

    print(f"{'C_A0':>6} {'switch_beta':>12}")
    for r in results:
        print(f"{r['C_A0']:6.1f} {r['switch_beta']:12.5f}")
    print(f"\nRange: {min(r['switch_beta'] for r in results):.3f} to "
          f"{max(r['switch_beta'] for r in results):.3f} "
          f"(ratio {max(r['switch_beta'] for r in results)/min(r['switch_beta'] for r in results):.2f}x) "
          f"across the paper's published C_A0 range alone.")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["C_A0"] for r in results], [r["switch_beta"] for r in results], "o-", color="#1b7837")
    ax.axvline(5.1, color="gray", ls=":", lw=1, label="nominal C_A0=5.1")
    ax.set_xlabel("C_A0 (mol/L)")
    ax.set_ylabel("switch beta*")
    ax.set_title("Active-constraint switch beta vs. feed concentration")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = "open_loop_testing/switch_beta_vs_feed_concentration.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    json_path = "open_loop_testing/switch_beta_vs_C_A0.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, out_path], data=None,
        params={"C_A0_values": C_A0_values, **kw}, script=__file__, start_time=start_time,
        extra={"results": results},
    )
    print(f"Saved {out_path}, {json_path}, {sidecar}")
