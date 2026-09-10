import json
import time

import numpy as np

from controllers.horizon_sweep import evaluate
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]

RTERM_F_GRID = [10.0, 30.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
PENALTY_GRID = [1e2, 5e2, 1e3]

TV_LIMIT = 50.0
NOMINAL_TV_LIMIT = 50.0
SETTLE_LIMIT = 60
NOMINAL_OFFSET_LIMIT = 0.5


def severe_disturbed(rterm_F, penalty_term_cons):
    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(base=FEED["C_A0"], magnitude=0.6, t_start=0.1)
    )
    return evaluate(n_horizon=20, beta=0.2, disturbance_fn=disturbed,
                     rterm_F=rterm_F, penalty_term_cons=penalty_term_cons)


def nominal(rterm_F, penalty_term_cons):
    return evaluate(n_horizon=20, beta=1.0,
                     rterm_F=rterm_F, penalty_term_cons=penalty_term_cons)


def passes(sev, nom):
    return (
        sev["F_total_variation"] < TV_LIMIT
        and nom["F_total_variation"] < NOMINAL_TV_LIMIT
        and nom["steps_to_settle"] is not None
        and nom["steps_to_settle"] <= SETTLE_LIMIT
        and nom["T_R_offset"] <= NOMINAL_OFFSET_LIMIT
    )


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    results = []

    print(f"{'rterm_F':>8} {'penalty':>8} | "
          f"{'SEV F TV':>9} {'SEV viol':>9} {'SEV off':>8} | "
          f"{'NOM F TV':>9} {'NOM off':>8} {'NOM settle':>10} {'NOM F off':>10} | pass")
    for rterm_F in RTERM_F_GRID:
        for penalty in PENALTY_GRID:
            sev = severe_disturbed(rterm_F, penalty)
            nom = nominal(rterm_F, penalty)
            ok = passes(sev, nom)
            results.append({"rterm_F": rterm_F, "penalty_term_cons": penalty,
                             "severe": sev, "nominal": nom, "passes": ok})
            settle = nom["steps_to_settle"]
            print(f"{rterm_F:8.0f} {penalty:8.0f} | "
                  f"{sev['F_total_variation']:9.2f} {sev['T_R_violation']:9.4f} "
                  f"{sev['T_R_offset']:8.4f} | "
                  f"{nom['F_total_variation']:9.2f} {nom['T_R_offset']:8.4f} "
                  f"{str(settle):>10} {nom['F_offset_final']:10.4f} | "
                  f"{'YES' if ok else '.'}")

    winners = [r for r in results if r["passes"]]
    print()
    if winners:
        best = min(winners, key=lambda r: (r["rterm_F"], r["penalty_term_cons"]))
        print(f"PICK: rterm_F={best['rterm_F']}, penalty_term_cons={best['penalty_term_cons']} "
              f"(severe TV {best['severe']['F_total_variation']:.2f}, "
              f"nominal settle {best['nominal']['steps_to_settle']} steps)")
    else:
        print("NO COMBINATION PASSES -- the grid does not contain a fix; widen it "
              "or the mechanism needs a different remedy (rate limit, not weights).")

    out_path = "controllers/chatter_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"rterm_F_grid": RTERM_F_GRID, "penalty_grid": PENALTY_GRID,
                "n_horizon": 20, "n_steps": 100,
                "thresholds": {"severe_TV": TV_LIMIT, "nominal_TV": NOMINAL_TV_LIMIT,
                                "settle_steps": SETTLE_LIMIT,
                                "nominal_offset": NOMINAL_OFFSET_LIMIT}},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {out_path}, {sidecar}")
