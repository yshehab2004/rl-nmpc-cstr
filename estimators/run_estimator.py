import numpy as np

from estimators.ekf import BetaEKF
from models.params import load_params
from simulation.measurement import measure
from simulation.simulator import build_simulator

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
T_STEP = PARAMS["nmpc"]["t_step"]


def run(x0, u, n_steps, ageing_fn=None, disturbance_fn=None, seed=None,
        noise=False, beta0=1.0, C_A0_known=True, dt=None, **ekf_kwargs):
    dt = T_STEP if dt is None else dt
    rng = np.random.default_rng(seed) if noise else None

    simulator = build_simulator(ageing_fn=ageing_fn, disturbance_fn=disturbance_fn)
    x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
    simulator.x0 = x0

    ekf = BetaEKF(x0.flatten(), beta0=beta0, **ekf_kwargs)
    u_col = np.asarray(u, dtype=float).reshape(-1, 1)

    history = {"t": [], "x_true": [], "x_hat": [], "beta_true": [], "beta_hat": [],
               "y": [], "C_A0_true": []}
    x = x0
    for k in range(n_steps):
        t_now = k * dt
        C_A0_true, T_in_true = (disturbance_fn(t_now) if disturbance_fn is not None
                                 else (FEED["C_A0"], FEED["T_in"]))
        beta_true = ageing_fn(t_now)[1] if ageing_fn is not None else 1.0

        C_A0_for_ekf = C_A0_true if C_A0_known else FEED["C_A0"]
        ekf.predict(u_col.flatten(), dt, C_A0=C_A0_for_ekf, T_in=T_in_true)

        x = simulator.make_step(u_col)
        y = measure(np.asarray(x).flatten(), rng=rng)
        ekf.update(y)

        history["t"].append(t_now)
        history["x_true"].append(np.asarray(x).flatten())
        history["x_hat"].append(ekf.x_hat)
        history["beta_true"].append(beta_true)
        history["beta_hat"].append(ekf.beta_hat)
        history["y"].append(y)
        history["C_A0_true"].append(C_A0_true)

    return {k: np.array(v) for k, v in history.items()}
