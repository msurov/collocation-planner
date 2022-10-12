from tkinter.tix import Tree
from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, fabs, ramp, sqrt
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from sympy import frac, hessian, jacobi
# from cartpend_dynamics import Parameters, Dynamics
from collocation import get_lgl_collocation_points, \
    get_lgl_weights, get_lagrange_basis


def solve_mech(d, E0 : float, ql : DM, qr : DM, deg : int, eps=1e-7):
    nq = deg + 1
    q = SX.sym('q', nq - 2)
    q = vertcat(ql, q, qr)

    F = 0

    for i in range(1, nq):
        qi = (q[i-1] + q[i]) / 2
        dqi = q[i] - q[i-1]

        Mi = substitute(d.M, d.q, qi)
        Ui = substitute(d.U, d.q, qi)
        tmp = fabs(dqi) * sqrt(Mi * ramp(E0 - Ui))
        F += tmp

    cost_function = F

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # for i in range(1, nq):
    #     qi = (q[i-1] + q[i]) / 2
    #     Ui = substitute(d.U, d.q, qi)
    #     eq = E0 - Ui
    #     constraints += [eq]
    #     constraints_lb += [1e-5 * np.ones(eq.shape)]
    #     constraints_ub += [1e+5 * np.ones(eq.shape)]
    
    # for i in range(1, nq):
    #     eq = q[i]
    #     constraints += [eq]
    #     constraints_lb += [-pi * np.ones(eq.shape)]
    #     constraints_ub += [pi * np.ones(eq.shape)]

    # nlp
    decision_variables = [reshape(q[1:-1], -1, 1)]
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
    dv0 = 1e-5*np.random.normal(size=decision_variables.shape)
    dv0 = DM(dv0)

    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)

    # tmp = substitute(constraints, decision_variables, sol['x'])
    # print(tmp)
    # exit()

    stat = BVP.stats()
    if not stat['success']:
        return None

    q_found = substitute(q, decision_variables, sol['x'])
    q_found = DM(q_found)
    q_found = np.array(q_found, float)
    t = np.linspace(0, 1, nq)

    return t, q_found


class Dynamics:
    def __init__(self) -> None:
        self.q = SX.sym('theta')
        self.dq = SX.sym('dtheta')
        self.K = self.dq**2/2
        self.U = cos(self.q)
        self.M = 1

def test():
    from cartpend_anim import CartPendAnim

    # nlinks = 2
    # p = Parameters(m_pend=0.15, l = 0.5, m_cart=0.1, g=9.8, nlinks=nlinks)
    # d = Dynamics(p)

    d = Dynamics()
    ql = DM([-pi/2])
    qr = DM([pi/2])

    E0 = 1.
    ans = None
    while ans is None:
        ans = solve_mech(d, E0, ql, qr, 25)
    
    t,q = ans
    print(t)
    print(q)

    if True:
        plt.plot(t, q)
        plt.show()

    if False:
        U = np.array([float(substitute(d.U, d.q, q[i])) for i in range(len(q))])
        plt.plot(t, E0 - U)
        plt.show()

    if False:
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

