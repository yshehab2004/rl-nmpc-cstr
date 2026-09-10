import numpy as np
import do_mpc

from models.cstr_model import build_model
from models.params import load_params

PARAMS = load_params()
_NMPC = PARAMS["nmpc"]
_BOUNDS_U = PARAMS["input_bounds"]
_BOUNDS_X = PARAMS["state_bounds"]
_FEED = PARAMS["nominal_feed"]
_UNCERTAIN = PARAMS["uncertain_params"]

_SCALE_F = 100.0
_SCALE_Q_DOT = 2000.0


def build_nmpc(target, n_horizon=None, t_step=None, beta_belief=None,
                model=None, penalty_term_cons=None,
                rterm_F=None, rterm_Q_dot=None, collocation_deg=None,
                feed=None):
    if model is None:
        model = build_model(for_controller=True)
    n_horizon = _NMPC["n_horizon"] if n_horizon is None else n_horizon
    t_step = _NMPC["t_step"] if t_step is None else t_step
    beta_belief = _UNCERTAIN["beta_nominal"] if beta_belief is None else beta_belief
    beta_fn = beta_belief if callable(beta_belief) else (lambda: beta_belief)
    penalty_term_cons = _NMPC["penalty_term_cons"] if penalty_term_cons is None else penalty_term_cons
    rterm_F = _NMPC["rterm_F"] if rterm_F is None else rterm_F
    rterm_Q_dot = _NMPC["rterm_Q_dot"] if rterm_Q_dot is None else rterm_Q_dot
    collocation_deg = _NMPC["collocation_deg"] if collocation_deg is None else collocation_deg

    feed_fn = feed if callable(feed) else (lambda: (feed if feed is not None
                                                    else (_FEED["C_A0"], _FEED["T_in"])))
    target_fn = target if callable(target) else (lambda t_now: target)

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.n_horizon = n_horizon
    mpc.settings.t_step = t_step
    mpc.settings.n_robust = _NMPC["n_robust"]
    mpc.settings.collocation_deg = collocation_deg
    mpc.settings.store_full_solution = False
    mpc.settings.supress_ipopt_output()

    T_R = model.x["T_R"]
    F = model.u["F"]
    T_R_ref = model.tvp["T_R_ref"]
    F_ref = model.tvp["F_ref"]

    track_T_R = ((T_R - T_R_ref) / 10.0) ** 2
    track_F = ((F - F_ref) / 10.0) ** 2
    mpc.set_objective(mterm=track_T_R, lterm=track_T_R + track_F)
    mpc.set_rterm(F=rterm_F, Q_dot=rterm_Q_dot)

    mpc.scaling["_u", "F"] = _SCALE_F
    mpc.scaling["_u", "Q_dot"] = _SCALE_Q_DOT

    mpc.bounds["lower", "_u", "F"] = _BOUNDS_U["F"]["lower"]
    mpc.bounds["upper", "_u", "F"] = _BOUNDS_U["F"]["upper"]
    mpc.bounds["lower", "_u", "Q_dot"] = _BOUNDS_U["Q_dot"]["lower"]
    mpc.bounds["upper", "_u", "Q_dot"] = _BOUNDS_U["Q_dot"]["upper"]

    mpc.bounds["lower", "_x", "C_a"] = _BOUNDS_X["C_a"]["lower"]
    mpc.bounds["upper", "_x", "C_a"] = _BOUNDS_X["C_a"]["upper"]
    mpc.bounds["lower", "_x", "C_b"] = _BOUNDS_X["C_b"]["lower"]
    mpc.bounds["upper", "_x", "C_b"] = _BOUNDS_X["C_b"]["upper"]
    mpc.bounds["lower", "_x", "T_R"] = _BOUNDS_X["T_R"]["lower"]
    mpc.bounds["lower", "_x", "T_K"] = _BOUNDS_X["T_K"]["lower"]

    mpc.set_nl_cons("T_R_upper", T_R, ub=_BOUNDS_X["T_R"]["upper_soft"],
                     soft_constraint=True, penalty_term_cons=penalty_term_cons)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        F_ref_now, T_R_ref_now = target_fn(np.asarray(t_now).item())
        beta_now = beta_fn()
        C_A0_now, T_in_now = feed_fn()
        for k in range(n_horizon + 1):
            tvp_template["_tvp", k, "F_ref"] = F_ref_now
            tvp_template["_tvp", k, "T_R_ref"] = T_R_ref_now
            tvp_template["_tvp", k, "beta_belief"] = beta_now
            tvp_template["_tvp", k, "C_A0_feed"] = C_A0_now
            tvp_template["_tvp", k, "T_in_feed"] = T_in_now
            tvp_template["_tvp", k, "C_b_set"] = _NMPC["C_b_setpoint_default"]
            tvp_template["_tvp", k, "d_T_R"] = 0.0
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.set_uncertainty_values(
        alpha=np.array([_UNCERTAIN["alpha_nominal"]]),
        beta=np.array([_UNCERTAIN["beta_nominal"]]),
    )

    mpc.setup()
    mpc.tvp_fun = tvp_fun
    return mpc


def run_closed_loop(mpc, simulator, x0, n_steps):
    x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    xs = [x0.flatten()]
    us = []
    x = x0
    for _ in range(n_steps):
        u = mpc.make_step(x)
        x = simulator.make_step(u)
        us.append(np.asarray(u).flatten())
        xs.append(np.asarray(x).flatten())

    t_step = mpc.settings.t_step
    return {
        "x": np.array(xs),
        "u": np.array(us),
        "t": np.arange(n_steps + 1) * t_step,
    }
