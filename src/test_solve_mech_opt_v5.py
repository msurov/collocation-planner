from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv, DM_inf, sum1
import scipy.special
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from sympy import frac, hessian, jacobi
from cartpend_dynamics import Parameters, Dynamics
from basis import get_basis
from numpy.polynomial import Polynomial
from scipy.optimize import root_scalar, brentq
from numpy.polynomial.legendre import legder, legroots, Legendre, legval


def solve_mech(d, E0 : float, ql : DM, qr : DM, deg : int, eps=1e-7):
    collocation_points = get_lgl_collocation_points(deg + 1)
    weights = get_lgl_weights(collocation_points)

    s = SX.sym('s')
    basis_fun = get_lagrange_basis(collocation_points)
    # basis_fun = get_basis(deg, 'Legendre')
    basis = basis_fun(s)
    D_basis = jacobian(basis, s)

    nq,_ = d.q.shape
    n,_ = basis.shape
    cq = SX.sym('cq', nq, n)
    q = cq @ basis
    dq = cq @ D_basis
    s1 = -1
    s2 = 1
    basis_left = substitute(basis, s, s1)
    basis_right = substitute(basis, s, s2)
    q_left = cq @ basis_left
    q_right = cq @ basis_right

    U_fun = Function('U', [d.q], [d.U])
    F = 0

    for wi,cp in zip(weights, collocation_points):
        qi = substitute(q, s, cp)
        Ui = U_fun(qi)
        F += wi * np.sqrt(E0 - Ui)

    cost_function = F

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # boundary conditions
    eq = q_left - ql
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = q_right - qr
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    # energy must be constant
    E_fun = Function('E', [d.q, d.dq], [d.K + d.U])
    for cp in collocation_points:
        qi = substitute(q, s, cp)
        dqi = substitute(dq, s, cp)
        eq = E_fun(qi, dqi) - E0
        constraints += [eq]
        constraints_lb += [-eps * DM.ones(eq.shape)]
        constraints_ub += [eps * DM.ones(eq.shape)]

    # nlp
    decision_variables = [reshape(cq, -1, 1)]
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
    dv0 = DM(dv0)

    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)

    q_found = substitute(q, decision_variables, sol['x'])
    q_func = Function('q', [s], [q_found])
    dq_found = substitute(dq, decision_variables, sol['x'])
    dq_func = Function('dq', [s], [dq_found])

    ss = np.linspace(s1, s2, 1000)
    qq = np.array([q_func(si) for si in ss], float)[:,:,0]
    dqq = np.array([dq_func(si) for si in ss], float)[:,:,0]

    return ss - s1, qq, dqq


def test():
    from cartpend_anim import CartPendAnim

    nlinks = 2
    p = Parameters(m_pend=0.15, l = 0.5, m_cart=0.1, g=9.8, nlinks=nlinks)
    d = Dynamics(p)

    ql = DM([0, pi-0.5, pi-0.1])
    qr = DM([0, pi+0.5, pi+0.5])
    t,q,dq = solve_mech(d, 20., ql, qr, 15)

    if True:
        E = np.zeros(len(t))
        E_fun = Function('E', [d.q, d.dq], [d.K + d.U])
        for i in range(len(t)): E[i] = E_fun(q[i], dq[i])
        plt.plot(t, E)
        plt.show()

    if False:
        anim = CartPendAnim('fig/cartpend.svg', nlinks)
        simdata = {
            't': t,
            'q': q
        }
        anim.run(simdata, filepath='data/anim.mp4')

    if False:
        rhs = Function('rhs', [d.q, d.dq], [substitute(d.rhs, d.u, 0)])
        n,_ = d.q.shape

        def f(_,st):
            dst = rhs(st[0:n], st[n:])
            return np.reshape(dst, -1)

        st0 = np.concatenate([q[0,:], dq[0,:]])
        sol = solve_ivp(f, [t[0], t[-1]], st0, max_step=1e-3)

        plt.gca().set_prop_cycle(None)
        plt.plot(sol.t, sol.y[0:n].T, '--', lw=2)
        plt.gca().set_prop_cycle(None)
        plt.plot(t, q)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    # test_lgl()
    test()

