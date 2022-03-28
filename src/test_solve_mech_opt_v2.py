from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv, DM_inf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cartpend_dynamics import Parameters, Dynamics
from cartpend_anim import CartPendAnim
from basis import get_basis, get_collocation_points, get_diap


def solve_mech_ivp(sys, q0, dq0, T, uoft):
    q = sys['q']
    dq = sys['dq']
    M = Function('M', [q], [sys['M']])
    C = Function('C', [q,dq], [sys['C']])
    G = Function('G', [q], [sys['G']])
    B = Function('B', [q], [sys['B']])
    u = sys['u']
    ddq = pinv(M(q)) @ (-C(q,dq) @ dq - G(q) + B(q) @ u)
    F = Function('F', [vertcat(q, dq), u], [vertcat(dq, ddq)])

    def rhs(t, st):
        u = uoft(t)
        dst = F(st, u)
        return np.reshape(dst, -1)

    st0 = np.reshape(vertcat(q0, dq0), -1)
    sol = solve_ivp(rhs, [0,T], st0, max_step=T / 1000)
    return sol

def solve_mechanical_opt(sys, ql, qr, umin, umax, deg, eps=1e-7):
    q = sys['q']
    u = sys['u']
    dq = sys['dq']
    M = Function('M', [q], [sys['M']])
    C = Function('C', [q,dq], [sys['C']])
    B = Function('B', [q], [sys['B']])
    G = Function('G', [q], [sys['G']])
    Bperp = Function('Bperp', [q], [sys['Bperp']])
    nq,_ = q.shape

    s = SX.sym('s')
    basis_fun = get_basis(deg, 'Legendre')
    basis = basis_fun(s)
    n,_ = basis.shape
    D_basis = jacobian(basis, s)
    D2_basis = jacobian(D_basis, s)
    s1, s2 = get_diap(basis_fun)
    collocation_points = get_collocation_points(basis_fun)
    basis_left = substitute(basis, s, s1)
    basis_right = substitute(basis, s, s2)
    D_basis_left = substitute(D_basis, s, s1)
    D_basis_right = substitute(D_basis, s, s2)

    cq = SX.sym('cq', nq, n)
    q = cq @ basis
    dq = cq @ D_basis
    d2q = cq @ D2_basis
    q_left = cq @ basis_left
    q_right = cq @ basis_right
    dq_left = cq @ D_basis_left
    dq_right = cq @ D_basis_right

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # differential equations
    Tinv_sq = SX.sym('Tinv_sq')
    constraints += [Tinv_sq]
    constraints_lb += [1e-4]
    constraints_ub += [1e+4]

    k = Tinv_sq * (s2 - s1)**2
    sys = Bperp(q) @ (
        k * (M(q) @ d2q + C(q,dq) @ dq) + G(q)
    )

    for cp in collocation_points:
        eq = substitute(sys, s, cp)
        constraints += [eq]
        constraints_lb += [-eps * DM.ones(eq.shape)]
        constraints_ub += [eps * DM.ones(eq.shape)]

    # control constraints
    Binv = pinv(B(q))
    k = (s2 - s1)**2 * Tinv_sq
    u_expr = Binv @ (M(q) @ d2q * k + C(q,dq) @ dq * k + G(q))

    for cp in collocation_points:
        ucp = substitute(u_expr, s, cp)
        constraints += [ucp]
        constraints_lb += [umin * DM.ones(ucp.shape)]
        constraints_ub += [umax * DM.ones(ucp.shape)]

    # boundary conditions
    eq = q_left - ql
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = q_right - qr
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = dq_left
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = dq_right
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    # generalized velocities
    for cp in collocation_points:
        eq = substitute(dq, s, cp)
        constraints += [eq]
        constraints_lb += [-10 * DM.ones(eq.shape)]
        constraints_ub += [10 * DM.ones(eq.shape)]

    # compose NLP
    decision_variables = [reshape(cq, -1, 1)]
    decision_variables += [Tinv_sq]
    cost_function = 1

    decision_variables = vertcat(*decision_variables)
    constraints = vertcat(*constraints)
    constraints_lb = vertcat(*constraints_lb)
    constraints_ub = vertcat(*constraints_ub)

    nlp = {
        'x': decision_variables,
        'f': cost_function,
        'g': constraints
    }

    # initial guess
    dv0 = substitute(decision_variables, cq, 1e-2*np.random.normal(size=cq.shape))
    dv0 = substitute(dv0, Tinv_sq, 1)
    dv0 = DM(dv0)

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)
    stat = BVP.stats()
    if not stat['success']:
        return None

    q_found = substitute(q, decision_variables, sol['x'])
    q_func = Function('q', [s], [q_found])
    Tinv_sq_found = float(substitute(Tinv_sq, decision_variables, sol['x']))
    T_found = 1 / np.sqrt(Tinv_sq_found)

    ss = np.linspace(s1, s2, 100)
    tt = (ss - s1) * T_found / (s2 - s1)
    qq = np.array([q_func(si) for si in ss], float)[:,:,0]

    return tt, qq


def test_solve_mech_opt():
    p = Parameters(m_pend=0.15, l = 0.5, m_cart=0.1, g=9.8, nlinks=3)
    d = Dynamics(p)
    sys = {
        'M': d.M,
        'C': d.C,
        'G': d.G,
        'B': d.B,
        'Bperp': d.Bperp,
        'q': d.q,
        'dq': d.dq,
        'u': d.u
    }
    ql = DM([0] + [pi] * p.nlinks)
    qr = DM([-2] + [0] * p.nlinks)

    ans = None
    while ans is None:
        ans = solve_mechanical_opt(sys, ql, qr, -10, 10, 15)

    t, q = ans
    simdata = {
        't': t,
        'q': q
    }
    anim = CartPendAnim('fig/cartpend.svg', nlinks=p.nlinks)
    anim.run(simdata, filepath='data/anim.mp4')


if __name__ == '__main__':
    test_solve_mech_opt()
