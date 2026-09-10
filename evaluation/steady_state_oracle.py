import casadi as ca

from models.params import load_params

PARAMS = load_params()
CERTAIN = PARAMS["certain_params"]


_MULTISTART_POINTS = (
    (5.0, -100.0), (5.0, -2000.0), (5.0, -6000.0), (5.0, -8000.0),
    (20.0, -3000.0), (40.0, -3000.0), (57.0, -1682.0), (70.0, -2000.0), (90.0, -3000.0),
)


def _solve_from(alpha, beta, C_A0, T_in, price_B, price_A0, price_energy, F0, Qd0):
    state_bounds = PARAMS["state_bounds"]
    input_bounds = PARAMS["input_bounds"]

    K0_ab, K0_bc, K0_ad = CERTAIN["K0_ab"], CERTAIN["K0_bc"], CERTAIN["K0_ad"]
    E_A_ab, E_A_bc, E_A_ad = CERTAIN["E_A_ab"], CERTAIN["E_A_bc"], CERTAIN["E_A_ad"]
    H_R_ab, H_R_bc, H_R_ad = CERTAIN["H_R_ab"], CERTAIN["H_R_bc"], CERTAIN["H_R_ad"]
    rho, Cp, Cp_k = CERTAIN["rho"], CERTAIN["Cp"], CERTAIN["Cp_k"]
    A_R, V_R, m_k, K_w = CERTAIN["A_R"], CERTAIN["V_R"], CERTAIN["m_k"], CERTAIN["K_w"]

    opti = ca.Opti()

    C_a = opti.variable()
    C_b = opti.variable()
    T_R = opti.variable()
    T_K = opti.variable()
    F = opti.variable()
    Q_dot = opti.variable()

    K_1 = beta * K0_ab * ca.exp(-E_A_ab / (T_R + 273.15))
    K_2 = K0_bc * ca.exp(-E_A_bc / (T_R + 273.15))
    K_3 = K0_ad * ca.exp(-alpha * E_A_ad / (T_R + 273.15))
    T_dif = T_R - T_K

    opti.subject_to(F * (C_A0 - C_a) - K_1 * C_a - K_3 * C_a ** 2 == 0)
    opti.subject_to(-F * C_b + K_1 * C_a - K_2 * C_b == 0)
    opti.subject_to(
        (K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * C_a ** 2 * H_R_ad) / (-rho * Cp)
        + F * (T_in - T_R)
        + (K_w * A_R) * (-T_dif) / (rho * Cp * V_R)
        == 0
    )
    opti.subject_to((Q_dot + K_w * A_R * T_dif) / (m_k * Cp_k) == 0)

    opti.subject_to(opti.bounded(state_bounds["C_a"]["lower"], C_a, state_bounds["C_a"]["upper"]))
    opti.subject_to(opti.bounded(state_bounds["C_b"]["lower"], C_b, state_bounds["C_b"]["upper"]))
    opti.subject_to(opti.bounded(state_bounds["T_R"]["lower"], T_R, state_bounds["T_R"]["upper_soft"]))
    opti.subject_to(opti.bounded(state_bounds["T_K"]["lower"], T_K, state_bounds["T_K"]["upper"]))
    opti.subject_to(opti.bounded(input_bounds["F"]["lower"], F, input_bounds["F"]["upper"]))
    opti.subject_to(opti.bounded(input_bounds["Q_dot"]["lower"], Q_dot, input_bounds["Q_dot"]["upper"]))

    profit = price_B * F * C_b - price_A0 * F * C_A0 - price_energy * (-Q_dot)
    opti.minimize(-profit)

    opti.set_initial(C_a, 0.8)
    opti.set_initial(C_b, 0.5)
    opti.set_initial(T_R, 134.14)
    opti.set_initial(T_K, 130.0)
    opti.set_initial(F, F0)
    opti.set_initial(Q_dot, Qd0)

    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0, "sb": "yes"})

    try:
        sol = opti.solve()
    except RuntimeError:
        return None

    return float(sol.value(C_b)), float(sol.value(profit))


def solve_steady_state_optimum(
    alpha: float,
    beta: float,
    C_A0: float,
    T_in: float,
    price_B: float,
    price_A0: float,
    price_energy: float,
):
    best = None
    for F0, Qd0 in _MULTISTART_POINTS:
        result = _solve_from(alpha, beta, C_A0, T_in, price_B, price_A0, price_energy, F0, Qd0)
        if result is not None and (best is None or result[1] > best[1]):
            best = result

    if best is None:
        raise RuntimeError(
            f"Steady-state oracle failed to converge from any start for alpha={alpha}, "
            f"beta={beta}, C_A0={C_A0}, T_in={T_in}"
        )

    return best


if __name__ == "__main__":
    feed = PARAMS["nominal_feed"]
    econ = PARAMS["economics"]
    uncertain = PARAMS["uncertain_params"]

    c_b_star, profit_star = solve_steady_state_optimum(
        alpha=uncertain["alpha_nominal"],
        beta=uncertain["beta_nominal"],
        C_A0=feed["C_A0"],
        T_in=feed["T_in"],
        price_B=econ["price_B"],
        price_A0=econ["price_A0"],
        price_energy=econ["price_energy"],
    )
    print(f"Nominal steady-state optimum: C_b* = {c_b_star:.4f} mol/L, profit* = {profit_star:.4f}")
