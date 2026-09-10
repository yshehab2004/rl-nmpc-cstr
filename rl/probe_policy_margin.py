import json
import time

import numpy as np

from models.params import load_params
from open_loop_testing.optimum_trajectory import find_optimum
from open_loop_testing.steady_state_map import solve_steady_state
from rl.policy_supervisor import RLSupervisor

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]
ECON = PARAMS["economics"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
ALPHA = PARAMS["uncertain_params"]["alpha_nominal"]
BETAS = np.round(np.linspace(0.15, 0.95, 17), 3)
CAMPAIGN_H = 75.0


def _on_manifold_context(beta, t_frac=0.5):
    F_grid = np.linspace(5.0, 35.0, 121)
    T_R_grid = np.linspace(100.0, 150.0, 201)
    opt = find_optimum(F_grid, T_R_grid, beta, ALPHA, FEED["C_A0"],
                        FEED["T_in"], C, **PRICES, T_R_max=T_LIMIT)
    F_s, T_R_s = opt["F_star"], opt["T_R_star"]
    C_a, C_b, T_K, Q_dot = solve_steady_state(F_s, T_R_s, beta, ALPHA,
                                               FEED["C_A0"], FEED["T_in"], C)
    return {"t": t_frac * CAMPAIGN_H, "beta_hat": float(beta),
            "C_A0": FEED["C_A0"], "T_in": FEED["T_in"],
            "x_hat": np.array([float(C_a), float(C_b), T_R_s, float(T_K)]),
            "u_last": np.array([F_s, float(Q_dot)])}


def probe(model_path, observe_cooling_margin=True):
    sup = RLSupervisor(model_path, campaign_h=CAMPAIGN_H,
                        observe_cooling_margin=observe_cooling_margin,
                        deterministic=True)
    rows = []
    for beta in BETAS:
        ctx = _on_manifold_context(beta)
        F_ref, T_R_ref, margin = sup.get_target(
            ctx["t"], beta_hat=ctx["beta_hat"], C_A0=ctx["C_A0"],
            T_in=ctx["T_in"], x_hat=ctx["x_hat"], u_last=ctx["u_last"])
        rows.append({"beta": float(beta), "F_ref": F_ref,
                      "T_R_ref": T_R_ref, "margin": margin})
    return rows


def summarise(rows):
    out = {}
    for key in ("F_ref", "T_R_ref", "margin"):
        v = np.array([r[key] for r in rows])
        d = np.diff(v)
        out[key] = {"min": float(v.min()), "max": float(v.max()),
                     "range": float(v.max() - v.min()),
                     "reversals": int(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0))}
    return out


if __name__ == "__main__":
    from pathlib import Path

    from utils.provenance import save_with_provenance

    start = time.time()
    results = {}
    for d in sorted(Path("rl/runs").iterdir()):
        if not (d / "model.zip").exists():
            continue
        meta = {}
        if (d / "train_meta.json").exists():
            meta = json.loads((d / "train_meta.json").read_text())
        w = meta.get("lambda_final") or meta.get("safety_penalty_weight")
        rows = probe(str(d / "model.zip"),
                      meta.get("observe_cooling_margin", True))
        results[d.name] = {"weight": w, "lagrangian": meta.get("lagrangian", False),
                            "rows": rows, "summary": summarise(rows)}
        print(f"\n=== {d.name}  (training weight {w}) ===", flush=True)
        print(f"{'beta':>6} {'F_ref':>8} {'T_R_ref':>9} {'margin':>8}")
        for r in rows:
            print(f"{r['beta']:6.2f} {r['F_ref']:8.2f} {r['T_R_ref']:9.2f} "
                  f"{r['margin']:8.3f}")
        s = results[d.name]["summary"]
        print(f"  ranges over beta: F {s['F_ref']['range']:.2f}, "
              f"T_R {s['T_R_ref']['range']:.2f}, margin {s['margin']['range']:.3f}")

    print("\n" + "=" * 74)
    print("MARGIN vs TRAINING PRICE  (does the agent buy back-off when it")
    print("becomes expensive to violate?)")
    print("=" * 74)
    print(f"{'policy':>32} {'weight':>10} {'mean margin':>12} {'margin range':>13}")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["weight"] or 0):
        mm = float(np.mean([x["margin"] for x in r["rows"]]))
        print(f"{name:>32} {r['weight'] or 0:10.0f} {mm:12.3f} "
              f"{r['summary']['margin']['range']:13.3f}")
    print("\nA flat margin column across a 4000x price range means the price")
    print("never reached the policy. A flat margin RANGE means the policy is")
    print("not state-dependent -- a learned constant, not an adaptive rule.")

    out = "rl/policy_margin_probe.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None, params={"betas": BETAS.tolist(), "on_manifold": True,
                                 "deterministic": True},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")
