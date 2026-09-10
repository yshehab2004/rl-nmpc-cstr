import json
import time

import numpy as np

from models.deactivation import make_deactivation_fn
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from rl.run_constrained_ceiling import ConstrainedOptimumSupervisor
from supervisors.campaign import run_campaign

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
Q_LOWER = PARAMS["input_bounds"]["Q_dot"]["lower"]
SENSOR_SIGMA = PARAMS["measurement"]["noise_frac"] * PARAMS["measurement"]["T_R_range"]
RL = PARAMS["rl"]

BETA_STAR = 0.385
BETAS = (0.25, 0.368, 0.60)
MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
CAMPAIGN_H = 6.0
STEP_AT_H = 2.0
STEP_MAG = 0.6
SEED = 20260907


def _run_one(job):
    beta, margin = job
    ageing_fn = make_deactivation_fn(beta_final=beta, campaign_h=1e-6)
    disturbance = build_feed_disturbance(
        C_A0_gen=step_disturbance(FEED["C_A0"], STEP_MAG, STEP_AT_H))
    t0 = time.time()
    run = run_campaign(ConstrainedOptimumSupervisor(margin),
                        campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                        disturbance_fn=disturbance, seed=SEED, n_horizon=20,
                        use_estimator=False,
                        decision_interval_h=RL["decision_interval_h"])
    T_R, Q = run["x"][:, 2], run["u"][:, 1]
    after = run["t"] >= STEP_AT_H
    viol = np.maximum(0.0, T_R - T_LIMIT)
    worst = int(np.argmax(viol)) if viol.max() > 0.0 else None
    return beta, margin, {
        "peak_T_R": float(T_R[after].max()),
        "peak_excursion": float(max(0.0, T_R[after].max() - T_LIMIT)),
        "violation_integral": run["metrics"]["violation_integral_degC_h"],
        "profit_per_hour": run["metrics"]["total_profit"] / CAMPAIGN_H,
        "cooling_saturated_at_peak": (None if worst is None
                                       else bool(Q[worst] <= Q_LOWER + 1.0)),
        "frac_cooling_saturated": float(np.mean(Q[after] <= Q_LOWER + 1.0)),
        "wall_s": time.time() - t0,
    }


def main():
    import multiprocessing as mp

    jobs = [(b, m) for b in BETAS for m in MARGINS]
    n = min(len(jobs), mp.cpu_count())
    print(f"{len(jobs)} campaigns on {n} workers", flush=True)
    out = {}
    with mp.Pool(processes=n) as pool:
        for beta, margin, row in pool.imap_unordered(_run_one, jobs):
            out.setdefault(f"beta_{beta}", {})[f"margin_{margin}"] = {
                "beta": beta, "margin": margin, **row}
            print(f"  beta={beta:.3f} m={margin:4.2f} "
                  f"exc={row['peak_excursion']:.4f} "
                  f"sat@peak={str(row['cooling_saturated_at_peak']):>5} "
                  f"profit/h={row['profit_per_hour']:7.1f} "
                  f"({row['wall_s']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start = time.time()
    results = main()

    print("\n" + "=" * 84)
    print(f"MARGIN LADDER ACROSS beta   (beta* = {BETA_STAR}, "
          f"sensor sigma {SENSOR_SIGMA:.2f} degC)")
    print("=" * 84)
    verdicts = {}
    for beta in BETAS:
        rows = [results[f"beta_{beta}"][f"margin_{m}"] for m in MARGINS]
        rel = "BELOW beta*" if beta < BETA_STAR else "ABOVE beta*"
        if abs(beta - 0.368) < 1e-9:
            rel = "at the changeover (ep45's beta)"
        print(f"\nbeta = {beta}  [{rel}]")
        print(f"{'margin':>7} {'excursion':>11} {'sat@peak':>9} {'profit/h':>10}")
        for r in rows:
            print(f"{r['margin']:7.2f} {r['peak_excursion']:11.4f} "
                  f"{str(r['cooling_saturated_at_peak']):>9} "
                  f"{r['profit_per_hour']:10.1f}")
        stuck = [r for r in rows if r["peak_excursion"] > 1e-9
                 and r["cooling_saturated_at_peak"]]
        freed = [r for r in rows if not r["cooling_saturated_at_peak"]]
        m_free = min((r["margin"] for r in freed), default=None)
        verdicts[beta] = {"plateau_len": len(stuck),
                           "margin_that_unsaturates": m_free}
        print(f"  -> margins stuck-while-saturated: {len(stuck)}; "
              f"cooling first unsaturated at margin "
              f"{m_free if m_free is not None else 'never'}")

    plateaus = {b: v["plateau_len"] for b, v in verdicts.items()}
    print("\n" + "-" * 84)
    if all(v >= 2 for v in plateaus.values()):
        print("Plateau at EVERY beta -> the mechanism is SATURATION.")
        print("beta* is incidental; the Chapter 6 claim is about actuator")
        print("authority, not about the constraint changeover.")
    elif plateaus.get(0.368, 0) >= 2 and all(
            plateaus.get(b, 0) < 2 for b in BETAS if abs(b - 0.368) > 1e-9):
        print("Plateau ONLY near beta* -> the mechanism is the CHANGEOVER.")
        print("The claim is about the constraint switching identity, and")
        print("beta* is the subject of the sentence.")
    else:
        print("Mixed. Plateau lengths per beta: "
              + ", ".join(f"{b}: {v}" for b, v in plateaus.items()))
        print("Neither clean claim is supported as stated -- report the")
        print("dependence on beta rather than a single mechanism.")

    out = "supervisors/margin_ladder.json"
    with open(out, "w") as f:
        json.dump({"results": results, "verdicts":
                   {str(k): v for k, v in verdicts.items()}}, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None,
        params={"betas": list(BETAS), "margins": list(MARGINS),
                "campaign_h": CAMPAIGN_H, "step_magnitude": STEP_MAG,
                "step_at_h": STEP_AT_H, "seed": SEED, "beta_star": BETA_STAR,
                "use_estimator": False},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")
