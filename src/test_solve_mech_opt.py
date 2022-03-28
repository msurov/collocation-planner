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

def solve_mechanical_opt(sys, q_constr_fun, umin, umax, deg, eps=1e-7):
    q = sys['q']
    u = sys['u']
    dq = sys['dq']
    M = Function('M', [q], [sys['M']])
    C = Function('C', [q,dq], [sys['C']])
    G = Function('G', [q], [sys['G']])
    B = Function('B', [q], [sys['B']])
    nq,_ = q.shape
    nu,_ = u.shape

    s = SX.sym('s')
    T = SX.sym('T')
    basis_fun = get_basis(deg)
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
    cu = SX.sym('cu', nu, n)
    u = cu @ basis
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
    constraints += [T]
    constraints_lb += [1e-2]
    constraints_ub += [1e+2]

    eq = M(q) @ d2q + C(q,dq) @ dq + (G(q) - B(q) @ u) * T**2 / (s2 - s1)**2

    for cp in collocation_points:
        tmp = substitute(eq, s, cp)
        constraints += [tmp]
        constraints_lb += [-eps * DM.ones(tmp.shape)]
        constraints_ub += [eps * DM.ones(tmp.shape)]

    # boundary conditions
    for cp in collocation_points:
        qcp = substitute(q, s, cp)
        eq = q_constr_fun(qcp)
        constraints += [eq]
        constraints_lb += [DM.zeros(eq.shape)]
        constraints_ub += [100000 * DM.ones(eq.shape)]

    eq = dq_left
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = dq_right
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    # control constraints
    for cp in collocation_points:
        tmp = substitute(u, s, cp)
        constraints += [tmp]
        constraints_lb += [umin * DM.ones(tmp.shape)]
        constraints_ub += [umax * DM.ones(tmp.shape)]

    tmp = substitute(u, s, s1)
    constraints += [tmp]
    constraints_lb += [umin * DM.ones(tmp.shape)]
    constraints_ub += [umax * DM.ones(tmp.shape)]

    tmp = substitute(u, s, s2)
    constraints += [tmp]
    constraints_lb += [umin * DM.ones(tmp.shape)]
    constraints_ub += [umax * DM.ones(tmp.shape)]

    # compose NLP
    decision_variables = [reshape(cq, -1, 1)]
    decision_variables += [reshape(cu, -1, 1)]
    decision_variables += [T]
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
    dv0 = substitute(decision_variables, T, 1)
    dv0 = substitute(dv0, cq, 1e-2*np.random.normal(size=cq.shape))
    dv0 = substitute(dv0, cu, 1e-2*np.random.normal(size=cu.shape))
    dv0 = DM(dv0)

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)
    values = DM(substitute(constraints, decision_variables, sol['x']))
    stat = BVP.stats()
    if not stat['success']:
        return None

    q_found = substitute(q, decision_variables, sol['x'])
    u_found = substitute(u, decision_variables, sol['x'])
    dq_found = substitute(dq, decision_variables, sol['x'])
    d2q_found = substitute(d2q, decision_variables, sol['x'])
    q_func = Function('Q', [s], [q_found])
    dq_func = Function('DQ', [s], [dq_found])
    u_func = Function('U', [s], [u_found])
    T_found = float(substitute(T, decision_variables, sol['x']))
    
    ss = np.linspace(s1, s2, 100)
    tt = np.array((ss - s1) / (s2 - s1) * T_found, float)
    qq = np.array([q_func(si) for si in ss], float)[:,:,0]
    uu =  np.array([u_func(si) for si in ss], float)[:,:,0]
    return tt, qq, uu


def test_solve_mech_opt():
    p = Parameters(m_pend=0.15, l = 0.5, m_cart=0.1, g=9.8, nlinks=2)
    d = Dynamics(p)
    sys = {
        'M': d.M,
        'C': d.C,
        'G': d.G,
        'B': d.B,
        'q': d.q,
        'dq': d.dq,
        'u': d.u
    }
    qmin = DM([-2, pi/2 - 0.8, pi/2 - 0.8])
    qmax = DM([2, pi/2 + 0.8, pi/2 + 0.8])
    q_constr_fun = Function('q_constr', [d.q], [vertcat(d.q - qmin, qmax - d.q)])

    ans = None
    while ans is None:
        ans = solve_mechanical_opt(sys, q_constr_fun, -100, 100, 9)

    t, q, u = ans

    plt.plot(t, u)
    plt.show()

    simdata = {
        't': t,
        'q': q
    }
    anim = CartPendAnim('fig/cartpend.svg', nlinks=p.nlinks)
    anim.run(simdata, filepath='data/anim.mp4', animtime=5)


if __name__ == '__main__':
    test_solve_mech_opt()
