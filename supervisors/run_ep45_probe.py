import json
import time

import numpy as np

from models.params import load_params
from rl.env import RandomAgeing
from rl.run_baselines import CAMPAIGN_H
from rl.run_constrained_ceiling import ConstrainedOptimumSupervisor
from rl.train import _training_feed
from supervisors.campaign import run_campaign

PARAMS = load_params()
RL = PARAMS["rl"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
Q_LOWER = PARAMS["input_bounds"]["Q_dot"]["lower"]
F_UPPER = PARAMS["input_bounds"]["F"]["upper"]
SENSOR_SIGMA = PARAMS["measurement"]["noise_frac"] * PARAMS["measurement"]["T_R_range"]

EPISODE = 45
MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def _run_one(margin):
    rng = np.random.default_rng(5000 + EPISODE)
    ageing_fn = RandomAgeing((RL["train_beta_final_low"],
                              RL["train_beta_final_high"]))(rng, CAMPAIGN_H)
    disturbance_fn = _training_feed()(rng, CAMPAIGN_H)
    t0 = time.time()
    run = run_campaign(ConstrainedOptimumSupervisor(margin),
                        campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                        disturbance_fn=disturbance_fn, seed=5000 + EPISODE,
                        n_horizon=20, use_estimator=False,
                        decision_interval_h=RL["decision_interval_h"])
    m = run["metrics"]
    T_R, F, Q = run["x"][:, 2], run["u"][:, 0], run["u"][:, 1]
    viol = np.maximum(0.0, T_R - T_LIMIT)
    worst = int(np.argmax(viol)) if viol.max() > 0.0 else None
    return margin, {
        "peak_T_R": float(T_R.max()),
        "peak_excursion": float(max(0.0, T_R.max() - T_LIMIT)),
        "violation_integral": m["violation_integral_degC_h"],
        "profit_per_hour": m["total_profit"] / CAMPAIGN_H,
        "t_at_peak": (None if worst is None else float(run["t"][worst])),
        "beta_at_peak": (None if worst is None else float(run["beta_true"][worst])),
        "F_at_peak": (None if worst is None else float(F[worst])),
        "Q_dot_at_peak": (None if worst is None else float(Q[worst])),
        "cooling_saturated_at_peak": (None if worst is None
                                       else bool(Q[worst] <= Q_LOWER + 1.0)),
        "F_at_bound_at_peak": (None if worst is None
                                else bool(F[worst] >= F_UPPER - 1e-3)),
        "frac_cooling_saturated": float(np.mean(Q <= Q_LOWER + 1.0)),
        "wall_s": time.time() - t0,
    }


def main():
    import multiprocessing as mp
    n = min(len(MARGINS), mp.cpu_count())
    print(f"episode {EPISODE}, {len(MARGINS)} margins on {n} workers", flush=True)
    out = {}
    with mp.Pool(processes=n) as pool:
        for margin, row in pool.imap_unordered(_run_one, MARGINS):
            out[f"margin_{margin}"] = {"margin": margin, **row}
            print(f"  m={margin:4.2f}  peak_T_R={row['peak_T_R']:7.3f}  "
                  f"excursion={row['peak_excursion']:.4f}  "
                  f"viol={row['violation_integral']:.5f}  "
                  f"profit/h={row['profit_per_hour']:7.1f}  "
                  f"cool_sat@peak={row['cooling_saturated_at_peak']}  "
                  f"({row['wall_s']:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start = time.time()
    results = main()

    print("\n" + "=" * 78)
    print(f"EPISODE {EPISODE}: peak excursion vs margin  "
          f"(sensor sigma {SENSOR_SIGMA:.2f} degC)")
    print("=" * 78)
    ms = sorted(results.values(), key=lambda r: r["margin"])
    print(f"{'margin':>7} {'peak T_R':>10} {'excursion':>11} {'profit/h':>10} "
          f"{'cool sat @peak':>15}")
    for r in ms:
        print(f"{r['margin']:7.2f} {r['peak_T_R']:10.3f} "
              f"{r['peak_excursion']:11.4f} {r['profit_per_hour']:10.1f} "
              f"{str(r['cooling_saturated_at_peak']):>15}")

    sat = [r for r in ms if r["cooling_saturated_at_peak"]]
    free = [r for r in ms if not r["cooling_saturated_at_peak"]]
    sat_exc = [r["peak_excursion"] for r in sat]
    free_exc = [r["peak_excursion"] for r in free]

    plateau = (len(sat) >= 3
               and max(sat_exc[1:], default=0.0) - min(sat_exc[1:], default=0.0)
               < 0.05)
    cliff = bool(free) and max(free_exc) < 1e-9
    m_free = min((r["margin"] for r in free), default=None)

    print(f"\nwhile cooling SATURATED  ({len(sat)} margins): "
          f"excursion {min(sat_exc):.4f} to {max(sat_exc):.4f}")
    if free:
        print(f"once cooling UNSATURATED ({len(free)} margins): "
              f"excursion {min(free_exc):.4f} to {max(free_exc):.4f}, "
              f"first at margin {m_free}")
    else:
        print("cooling never came off the floor in this ladder")

    if plateau and cliff:
        print("\n  -> THREE REGIMES. Margin is INERT while the cooler is")
        print("     saturated -- more back-off is bought and nothing is")
        print("     delivered, because the actuator is already at its stop --")
        print(f"     and works DISCONTINUOUSLY at margin {m_free}, where the")
        print("     cooling comes off the floor and the excursion goes to zero.")
        print("     That is a mechanism, and it is what Chapter 6 explains.")
    elif not free:
        print("\n  -> Cooling saturated throughout; this ladder cannot see the")
        print("     transition. Extend the margin range before concluding.")
    else:
        print("\n  -> Regimes not cleanly separated. Report the excursion and")
        print("     the saturation flag together rather than claiming a")
        print("     mechanism this ladder does not show.")

    out = "supervisors/ep45_margin_probe.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None,
        params={"episode": EPISODE, "margins": list(MARGINS),
                "campaign_h": CAMPAIGN_H, "seed": 5000 + EPISODE,
                "use_estimator": False},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")
