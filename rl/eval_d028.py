import json
import sys
import time
from pathlib import Path

import numpy as np

from open_loop_testing.optimum_trajectory import find_optimum
from rl.probe_policy_margin import (ALPHA, BETAS, C, CAMPAIGN_H, FEED,
                                     PRICES, T_LIMIT, _on_manifold_context)
from rl.policy_supervisor import RLSupervisor

PREREG = {
    "P1_baseline_mean_abs_dF_low_beta": 8.27,
    "P1_threshold": 2.0,
    "P2_legacy_pinned": 9, "P2_legacy_total": 24,
    "P2_threshold_fraction_of_40": 2,
    "margin_ceiling": 5.0,
    "reference_by_regime": {"high": 0.49, "mid": 3.80, "low": 8.27},
}
LOW = 0.30
HIGH = 0.60

_LOW_BETAS = [b for b in BETAS if b <= LOW]
assert len(_LOW_BETAS) == 4, (
    f"D-028's denominators assume 4 grid points at beta<={LOW}; "
    f"BETAS gives {len(_LOW_BETAS)}: {_LOW_BETAS}. The pre-registered "
    f"'9 of 24' and 'fewer than 2 of 40' no longer mean what they say.")


def probe_with_optimum(model_path, observe_cooling_margin=True):
    sup = RLSupervisor(model_path, campaign_h=CAMPAIGN_H,
                        observe_cooling_margin=observe_cooling_margin,
                        deterministic=True)
    F_grid = np.linspace(5.0, 35.0, 121)
    T_R_grid = np.linspace(100.0, 150.0, 201)
    rows = []
    for beta in BETAS:
        ctx = _on_manifold_context(beta)
        F_ref, T_R_ref, margin = sup.get_target(
            ctx["t"], beta_hat=ctx["beta_hat"], C_A0=ctx["C_A0"],
            T_in=ctx["T_in"], x_hat=ctx["x_hat"], u_last=ctx["u_last"])
        opt = find_optimum(F_grid, T_R_grid, beta, ALPHA, FEED["C_A0"],
                            FEED["T_in"], C, **PRICES, T_R_max=T_LIMIT)
        rows.append({"beta": float(beta),
                      "F_ref": float(F_ref), "F_star": float(opt["F_star"]),
                      "dF": float(F_ref - opt["F_star"]),
                      "T_R_ref": float(T_R_ref),
                      "T_R_star": float(opt["T_R_star"]),
                      "margin": float(margin)})
    return rows


def by_regime(all_rows):
    out = {}
    for name, keep in (("high", lambda b: b > HIGH),
                        ("mid", lambda b: HIGH >= b > LOW),
                        ("low", lambda b: b <= LOW)):
        v = [abs(r["dF"]) for r in all_rows if keep(r["beta"])]
        out[name] = {"mean_abs_dF": float(np.mean(v)) if v else None,
                      "max_abs_dF": float(np.max(v)) if v else None,
                      "n": len(v)}
    return out


def main(runs_dir="rl/runs_d028/A1_new_withmargin"):
    t_start = time.time()
    d = Path(runs_dir)
    policies = sorted(p for p in d.iterdir() if (p / "model.zip").exists())
    if not policies:
        sys.exit(f"no models under {d}")
    print(f"D-028 predictions 1 and 2 -- arm {d.name}, {len(policies)} seeds\n",
          flush=True)

    per_seed, all_rows, ceiling_hits, low_probes = {}, [], 0, 0
    for p in policies:
        meta = json.loads((p / "train_meta.json").read_text())
        t0 = time.time()
        rows = probe_with_optimum(str(p / "model.zip"),
                                   meta.get("observe_cooling_margin", True))
        ceiling = meta.get("margin_hi")
        assert ceiling is not None, f"{p.name} has no margin_hi in train_meta"
        low = [r for r in rows if r["beta"] <= LOW]
        hits = sum(1 for r in low if r["margin"] >= ceiling - 1e-6)
        ceiling_hits += hits
        low_probes += len(low)
        all_rows.extend(rows)
        per_seed[p.name] = {"rows": rows, "by_regime": by_regime(rows),
                             "margin_ceiling": ceiling,
                             "low_beta_ceiling_hits": hits,
                             "wall_s": time.time() - t0}
        r = per_seed[p.name]["by_regime"]
        print(f"  {p.name[-26:]:<26} "
              f"|dF| low {r['low']['mean_abs_dF']:6.2f}  "
              f"mid {r['mid']['mean_abs_dF']:6.2f}  "
              f"high {r['high']['mean_abs_dF']:5.2f}   "
              f"ceiling {hits}/{len(low)}  ({per_seed[p.name]['wall_s']:.0f}s)",
              flush=True)

    pooled = by_regime(all_rows)
    seed_low = [v["by_regime"]["low"]["mean_abs_dF"] for v in per_seed.values()]
    p1_pass = pooled["low"]["mean_abs_dF"] < PREREG["P1_threshold"]
    p2_pass = ceiling_hits < PREREG["P2_threshold_fraction_of_40"]

    print("\n" + "=" * 72)
    print("PREDICTION 1 (PRIMARY): mean |dF| at beta <= 0.30 below 2.0")
    print("=" * 72)
    print(f"  D-028 baseline (6 legacy policies) : {PREREG['P1_baseline_mean_abs_dF_low_beta']:.2f}")
    print(f"  {d.name}, pooled over {len(policies)} seeds : "
          f"{pooled['low']['mean_abs_dF']:.2f}")
    print(f"  per-seed spread                    : "
          f"{min(seed_low):.2f} to {max(seed_low):.2f}")
    print(f"  VERDICT: {'PASS' if p1_pass else 'FAIL'} "
          f"(threshold {PREREG['P1_threshold']})")
    print("\n  by regime, against D-028's reference:")
    for k, ref in (("high", 0.49), ("mid", 3.80), ("low", 8.27)):
        print(f"    {k:>4}: {pooled[k]['mean_abs_dF']:6.2f}   "
              f"(was {ref:.2f}, n={pooled[k]['n']})")

    print("\n" + "=" * 72)
    print("PREDICTION 2: fewer than 2 of 40 low-beta probes pin at the ceiling")
    print("=" * 72)
    print(f"  legacy  : {PREREG['P2_legacy_pinned']} of {PREREG['P2_legacy_total']}")
    print(f"  {d.name}: {ceiling_hits} of {low_probes}")
    print(f"  VERDICT: {'PASS' if p2_pass else 'FAIL'} "
          f"(threshold fewer than {PREREG['P2_threshold_fraction_of_40']})")
    print("\nPredictions 3 and 4 need arm A3 and the S1/S2/S3 scenarios.")

    payload = {"arm": d.name, "n_seeds": len(policies), "prereg": PREREG,
                "pooled_by_regime": pooled, "per_seed": per_seed,
                "P1_pass": bool(p1_pass), "P2_pass": bool(p2_pass),
                "P1_value": pooled["low"]["mean_abs_dF"],
                "P2_ceiling_hits": ceiling_hits, "P2_low_probes": low_probes}
    out = f"rl/eval_d028_{d.name}.json"
    Path(out).write_text(json.dumps(payload, indent=2))
    from utils.provenance import save_with_provenance
    save_with_provenance(out, None,
                          {"runs_dir": str(d), "n_seeds": len(policies),
                           "betas_low": [float(b) for b in _LOW_BETAS],
                           "prereg": PREREG},
                          script="rl/eval_d028.py", start_time=t_start)
    print(f"\nwrote {out} and its provenance sidecar")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rl/runs_d028/A1_new_withmargin")
