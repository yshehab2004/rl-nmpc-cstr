import numpy as np

from controllers.nmpc import build_nmpc
from estimators.ekf import BetaEKF
from models.params import load_params
from open_loop_testing.optimum_trajectory import compute_profit
from open_loop_testing.target_optimizer import solve_target
from simulation.measurement import measure
from simulation.simulator import build_simulator, initial_state

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]
ECON = PARAMS["economics"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])
T_STEP = PARAMS["nmpc"]["t_step"]
DECISION_INTERVAL_H = PARAMS["rl"]["decision_interval_h"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
ALPHA = PARAMS["uncertain_params"]["alpha_nominal"]

_HIST_KEYS = ("t", "x", "u", "beta_true", "beta_hat", "C_A0", "T_in",
              "F_ref", "T_R_ref", "margin", "F_s", "T_R_s")


class CampaignEngine:

    def __init__(self, ageing_fn=None, disturbance_fn=None, seed=None,
                  noise=True, use_estimator=True, n_horizon=None, x0=None):
        self.ageing_fn = ageing_fn
        self.disturbance_fn = disturbance_fn
        self.use_estimator = use_estimator
        self.n_horizon = n_horizon
        self.rng = np.random.default_rng(seed) if noise else None

        self.x0 = (initial_state() if x0 is None
                   else np.asarray(x0, dtype=float).reshape(-1, 1))
        self.simulator = build_simulator(ageing_fn=ageing_fn,
                                          disturbance_fn=disturbance_fn)
        self.simulator.x0 = self.x0
        self.ekf = BetaEKF(self.x0.flatten(), beta0=1.0,
                            noise_frac=None if noise else 0.0)

        self.holder = {"target": None, "t": 0.0, "beta": 1.0}

        self.mpc = None
        self.x = self.x0
        self.u_last = None
        self.k = 0
        self.decisions = []
        self.hist = {key: [] for key in _HIST_KEYS}


    @property
    def t_now(self):
        return self.k * T_STEP

    def feed_at(self, t_now):
        if self.disturbance_fn is None:
            return FEED["C_A0"], FEED["T_in"]
        return self.disturbance_fn(t_now)

    def beta_at(self, t_now):
        return 1.0 if self.ageing_fn is None else self.ageing_fn(t_now)[1]

    def _belief(self):
        return self.ekf.beta_hat if self.use_estimator else self.beta_at(self.t_now)


    def context(self):
        C_A0, T_in = self.feed_at(self.t_now)
        return {"t": self.t_now, "beta_hat": self._belief(),
                "C_A0": C_A0, "T_in": T_in,
                "x_hat": self.ekf.x_hat, "u_last": self.u_last}

    def set_target(self, F_ref, T_R_ref, margin):
        C_A0_meas, T_in_meas = self.feed_at(self.t_now)
        beta = self._belief()
        self.holder["beta"] = beta
        projected = solve_target(F_ref, T_R_ref, margin, beta, C_A0_meas,
                                  alpha=ALPHA, T_in=T_in_meas, C=C)
        self.holder["target"] = (projected["F_s"], projected["T_R_s"])
        decision = {"t": self.t_now, "F_ref": F_ref, "T_R_ref": T_R_ref,
                     "margin": margin, "F_s": projected["F_s"],
                     "T_R_s": projected["T_R_s"], "beta_used": beta,
                     "C_A0": C_A0_meas, "T_in": T_in_meas}
        self.decisions.append(decision)
        return decision

    def _ensure_mpc(self):
        if self.mpc is not None:
            return
        self.mpc = build_nmpc(target=lambda t_now: self.holder["target"],
                               beta_belief=lambda: self.holder["beta"],
                               feed=lambda: self.feed_at(self.holder["t"]),
                               n_horizon=self.n_horizon)
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()

    def advance(self, n_steps):
        if self.holder["target"] is None:
            raise RuntimeError(
                "advance() before set_target(): the NMPC has nothing to track. "
                "The cycle is context -> set_target -> advance.")
        self._ensure_mpc()
        decision = self.decisions[-1]

        for _ in range(n_steps):
            t_now = self.t_now
            self.holder["t"] = t_now
            C_A0_true, T_in_true = self.feed_at(t_now)

            u = self.mpc.make_step(self.x)
            self.ekf.predict(u, T_STEP, C_A0=C_A0_true, T_in=T_in_true)
            self.x = self.simulator.make_step(u)
            self.ekf.update(measure(np.asarray(self.x).flatten(), rng=self.rng))
            self.u_last = np.asarray(u).flatten()

            self.hist["t"].append(t_now)
            self.hist["x"].append(np.asarray(self.x).flatten())
            self.hist["u"].append(self.u_last)
            self.hist["beta_true"].append(self.beta_at(t_now))
            self.hist["beta_hat"].append(self.ekf.beta_hat)
            self.hist["C_A0"].append(C_A0_true)
            self.hist["T_in"].append(T_in_true)
            self.hist["F_ref"].append(decision["F_ref"])
            self.hist["T_R_ref"].append(decision["T_R_ref"])
            self.hist["margin"].append(decision["margin"])
            self.hist["F_s"].append(decision["F_s"])
            self.hist["T_R_s"].append(decision["T_R_s"])
            self.k += 1

    def result(self):
        out = {key: np.array(val) for key, val in self.hist.items()}
        out["decisions"] = self.decisions
        out["metrics"] = summarize(out)
        return out


def run_campaign(supervisor, campaign_h, ageing_fn=None, disturbance_fn=None,
                  seed=None, noise=True, use_estimator=True,
                  decision_interval_h=None, n_horizon=None, x0=None):
    decision_interval_h = (DECISION_INTERVAL_H if decision_interval_h is None
                           else decision_interval_h)
    n_steps = int(round(campaign_h / T_STEP))
    steps_per_decision = max(1, int(round(decision_interval_h / T_STEP)))

    engine = CampaignEngine(ageing_fn=ageing_fn, disturbance_fn=disturbance_fn,
                             seed=seed, noise=noise, use_estimator=use_estimator,
                             n_horizon=n_horizon, x0=x0)

    while engine.k < n_steps:
        ctx = engine.context()
        F_ref, T_R_ref, margin = supervisor.get_target(
            ctx["t"], beta_hat=ctx["beta_hat"], C_A0=ctx["C_A0"],
            T_in=ctx["T_in"], x_hat=ctx["x_hat"], u_last=ctx["u_last"])
        engine.set_target(F_ref, T_R_ref, margin)
        engine.advance(min(steps_per_decision, n_steps - engine.k))

    out = engine.result()
    out["n_supervisor_solves"] = getattr(supervisor, "n_solves",
                                          len(engine.decisions))
    return out


def summarize(run):
    T_R = run["x"][:, 2]
    C_b = run["x"][:, 1]
    F, Q_dot = run["u"][:, 0], run["u"][:, 1]

    profit = compute_profit(F, C_b, run["C_A0"], Q_dot, **PRICES)
    violation = np.maximum(0.0, T_R - T_LIMIT)

    return {
        "total_profit": float(profit.sum() * T_STEP),
        "mean_profit": float(profit.mean()),
        "violation_max": float(violation.max()),
        "violation_integral_degC_h": float(violation.sum() * T_STEP),
        "steps_above_limit": int((violation > 0).sum()),
        "fraction_time_violating": float((violation > 0).mean()),
        "T_R_max": float(T_R.max()),
        "mean_F": float(F.mean()),
        "mean_C_b": float(C_b.mean()),
        "beta_hat_max_error": float(np.abs(run["beta_hat"] - run["beta_true"]).max()),
    }
