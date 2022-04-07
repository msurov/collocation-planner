from ast import Param
from casadi import SX, DM, cos, sin, jacobian, vertcat, pinv, Function, \
    cumsum, horzcat, jtimes, sumsqr, sum1, sum2, substitute, DM_eye
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
        U = p.m_pend * p.g * p.l * cos(theta)
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


def kinetic_energy_form(K, dq):
    p = jacobian(K, dq)
    M = jacobian(p, dq)
    return M

def coriolis_mat(M, q, dq):
    Z = jacobian(M @ dq, q)
    C = Z - 0.5 * Z.T
    return C


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
        self.u = u

        positions = self.get_positions(p)
        velocities = jtimes(positions, q, dq)
        velocities_sq = sum2(velocities**2)
        K = (p.m_cart * velocities_sq[0] + p.m_pend * sum1(velocities_sq[1:])) / 2
        U = p.m_cart * p.g * positions[0,1] + \
            p.m_pend * p.g * sum1(positions[1:,1])
        M = kinetic_energy_form(K, dq)
        C = coriolis_mat(M, q, dq)
        G = jacobian(U, q).T
        B = vertcat(1, SX.zeros(self.nlinks,1))
        Bperp = DM_eye(self.nlinks + 1)[1:,:]
        self.M = M
        self.C = C
        self.G = G
        self.B = B
        self.Bperp = Bperp
        self.U = U

        ddq = pinv(M) @ (-C @ dq - G + B @ u)
        self.rhs = vertcat(dq, ddq)


    def get_positions(self, p : Parameters):
        x = self.q[0]
        theta = self.q[1:]
        sin_theta = sin(theta)
        cos_theta = cos(theta)
        px = cumsum(vertcat(x, p.l * sin_theta))
        py = cumsum(vertcat(0, p.l * cos_theta))
        p = horzcat(px, py)
        return p


def test():
    from cartpend_anim import CartPendAnim

    nlinks = 5
    p = Parameters(m_pend = 0.1, m_cart=0.5, l = 0.5, g = 9.8, nlinks=nlinks)
    d = Dynamics(p)

    F = Function('rhs', [vertcat(d.q, d.dq), d.u], [d.rhs])
    def rhs(t, st):
        u = 0
        dst = F(st, u)
        return np.reshape(dst, (-1))
    
    st0 = 1e-3 * np.random.normal(size=2 * (nlinks + 1))
    tspan = [0, 10]
    sol = solve_ivp(rhs, tspan, st0, max_step=1e-3)

    anim = CartPendAnim('fig/cartpend.svg', nlinks)
    simdata = {
        't': sol.t,
        'q': sol.y[0:nlinks+1].T
    }

    anim.run(simdata, filepath='data/anim.mp4')


if __name__ == '__main__':
    test()
