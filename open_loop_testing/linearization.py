import numpy as np


def ode_rhs(x, u, alpha, beta, C_A0, T_in, C):
    C_a, C_b, T_R, T_K = x
    F, Q_dot = u
    T = T_R + 273.15
    K1 = beta * C["K0_ab"] * np.exp(-C["E_A_ab"] / T)
    K2 = beta * C["K0_bc"] * np.exp(-C["E_A_bc"] / T)
    K3 = C["K0_ad"] * np.exp(-alpha * C["E_A_ad"] / T)

    dC_a = F * (C_A0 - C_a) - K1 * C_a - K3 * C_a ** 2
    dC_b = -F * C_b + K1 * C_a - K2 * C_b
    T_dif = T_R - T_K
    dT_R = (
        (K1 * C_a * C["H_R_ab"] + K2 * C_b * C["H_R_bc"] + K3 * C_a ** 2 * C["H_R_ad"]) / (-C["rho"] * C["Cp"])
        + F * (T_in - T_R)
        + ((C["K_w"] * C["A_R"]) * (-T_dif)) / (C["rho"] * C["Cp"] * C["V_R"])
    )
    dT_K = (Q_dot + C["K_w"] * C["A_R"] * T_dif) / (C["m_k"] * C["Cp_k"])
    return np.array([dC_a, dC_b, dT_R, dT_K])


def linearize(x_ss, u_ss, alpha, beta, C_A0, T_in, C, eps=1e-6):
    n, m = len(x_ss), len(u_ss)
    kw = dict(alpha=alpha, beta=beta, C_A0=C_A0, T_in=T_in, C=C)

    A = np.zeros((n, n))
    for i in range(n):
        h = eps * max(abs(x_ss[i]), 1.0)
        x_plus, x_minus = x_ss.copy(), x_ss.copy()
        x_plus[i] += h
        x_minus[i] -= h
        A[:, i] = (ode_rhs(x_plus, u_ss, **kw) - ode_rhs(x_minus, u_ss, **kw)) / (2 * h)

    B = np.zeros((n, m))
    for j in range(m):
        h = eps * max(abs(u_ss[j]), 1.0)
        u_plus, u_minus = u_ss.copy(), u_ss.copy()
        u_plus[j] += h
        u_minus[j] -= h
        B[:, j] = (ode_rhs(x_ss, u_plus, **kw) - ode_rhs(x_ss, u_minus, **kw)) / (2 * h)

    return A, B
