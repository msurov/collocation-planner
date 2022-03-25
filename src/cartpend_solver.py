from casadi import MX,SX,DM,polyval,vertcat,substitute,\
    Function,reshape,nlpsol,sin, pi
from scipy.integrate import solve_ivp
from cartpend_dynamics import Dynamics, Parameters
from bvp_solver import solve_bvp, solve_control, solve_mechanical_bvp
import matplotlib.pyplot as plt
import numpy as np
from cartpend_anim import CartPendAnim


def test():
    p = Parameters(m_pend = 0.1, m_cart=0.5, l = 0.5, g = 9.8)
    d = Dynamics(p)

    st = vertcat(d.q, d.dq)
    rhs = Function('rhs', [st, d.u], [d.rhs])
    bc_left = vertcat(
        d.q - vertcat(0, pi),
        d.dq
    )
    bc_right = vertcat(
        d.q - vertcat(0, 0),
        d.dq
    )
    bc_left_fun = Function('bc_left', [st], [bc_left])
    bc_right_fun = Function('bc_right', [st], [bc_right])

    stpoly,upoly,T = solve_control(rhs, bc_left_fun, bc_right_fun, 15)

    s = np.linspace(0, 1, 100)
    x = np.polyval(stpoly[:,0], s)
    theta = np.polyval(stpoly[:,1], s)
    u = np.polyval(upoly, s)

    def f(t, st):
        u = np.polyval(upoly, t/T)
        dst = rhs(st, u)
        return np.reshape(dst, (-1,))

    tspan = [0,T]
    st0 = np.array([x[0], theta[0], 0, 0])
    sol = solve_ivp(f, tspan, st0, max_step=1e-4)

    _,axes = plt.subplots(3, 1, sharex=True)
    plt.sca(axes[0])
    plt.grid(True)
    plt.plot(sol.t, sol.y[0], '--', lw=2)
    plt.plot(s * T, x)
    plt.sca(axes[1])
    plt.grid(True)
    plt.plot(sol.t, sol.y[1], '--', lw=2)
    plt.plot(s * T, theta)
    plt.sca(axes[2])
    plt.grid(True)
    plt.plot(s * T, u)

    plt.show()


def test2():
    p = Parameters(m_pend = 0.1, m_cart=0.5, l = 0.5, g = 9.8, nlinks=1)
    d = Dynamics(p)

    sys = {
        'M': Function('M', [d.q], [d.M]),
        'C': Function('C', [d.q, d.dq], [d.C]),
        'G': Function('G', [d.q], [d.G]),
        'B': Function('B', [d.q], [d.B]),
        'q': d.q,
        'dq': d.dq,
        'u': d.u,
    }

    ql = DM([0, pi])
    qr = DM([0, 0])
    qpoly, upoly, T = solve_mechanical_bvp(sys, ql, qr, -50, 50, deg=13)

    s = np.linspace(0, 1, 100)
    x = np.polyval(qpoly[:,0], s)
    theta = np.polyval(qpoly[:,1], s)
    u = np.polyval(upoly, s)

    anim = CartPendAnim('fig/cartpend.svg', p.nlinks)
    simdata = {
        't': s * T,
        'q': np.array([x, theta]).T
    }
    anim.run(simdata, animtime=2, filepath='data/anim.mp4')

    _,axes = plt.subplots(3, 1, sharex=True)
    plt.sca(axes[0])
    plt.grid(True)
    plt.plot(s * T, x)
    plt.sca(axes[1])
    plt.grid(True)
    plt.plot(s * T, theta)
    plt.sca(axes[2])
    plt.grid(True)
    plt.plot(s * T, u)
    plt.show()


if __name__ == '__main__':
    test2()
