from ast import Param
from casadi import SX, DM, cos, sin, jacobian, vertcat, pinv, Function
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


@dataclass
class Parameters:
    m_pend : float # pendulum mass
    l : float # pendulum length
    m_cart : float # cart mass
    g : float # gravity accel
    nlinks : int

class DynamicsOld:

    def __init__(self, p : Parameters):
        q = SX.sym('q', 2)
        dq = SX.sym('dq', 2)
        u = SX.sym('u')
        theta = q[1]

        M = SX.zeros((2,2))
        M[0,0] = p.m_pend + p.m_cart
        M[1,0] = M[0,1] = p.m_pend * p.l * cos(theta)
        M[1,1] = p.m_pend * p.l**2
        Z = jacobian(M @ dq, q)
        C = Z - 0.5 * Z.T
        U = p.m_pend * p.l * cos(theta)
        G = jacobian(U, q).T
        B = DM([[1], 0])
        ddq = pinv(M) @ (-C @ dq - G + B @ u)
        rhs = vertcat(dq, ddq)

        self.M = M
        self.C = C
        self.G = G
        self.B = B
        self.q = q
        self.dq = dq
        self.ddq = ddq
        self.u = u
        self.rhs = rhs


class Dynamics:

    def __init__(self, p : Parameters):
        assert p.nlinks >= 1

        self.nlinks = p.nlinks

        q = SX.sym('q', self.nlinks + 1)
        dq = SX.sym('dq', self.nlinks + 1)
        u = SX.sym('u')
        theta = q[1]

        self.q = q
        self.dq = dq

        self.get_positions()


    def get_positions(self):
        thetas = self.q[1:]
        sin(thetas)
        cos(thetas)


def test():
    p = Parameters(m_pend = 0.1, m_cart=0.5, l = 0.5, g = 9.8, nlinks=2)
    d = Dynamics(p)

    exit()

    F = Function('rhs', [vertcat(d.q, d.dq), d.u], [d.rhs])
    def rhs(t, st):
        u = -(st[0] - 1) * 5
        dst = F(st, u)
        return np.reshape(dst, (-1))
    
    st0 = np.zeros(4)
    tspan = [0, 10]
    sol = solve_ivp(rhs, tspan, st0, max_step=1e-3)
    plt.plot(sol.t, sol.y[0])
    plt.plot(sol.t, sol.y[1])
    plt.show()


if __name__ == '__main__':
    test()
