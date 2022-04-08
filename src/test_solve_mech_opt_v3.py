from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv, DM_inf, sum1
import scipy.special
import numpy as np
from cartpend_dynamics import Parameters, Dynamics


def get_lagrange_basis(points):
    x = SX.sym('x')
    n = len(points)
    basis = []

    for i in range(n):
        b = 1
        for j in range(n):
            if i == j: continue
            b *= (x - points[j]) / (points[i] - points[j])
        basis += [b]
    
    basis = vertcat(*basis)
    return Function('Lagrange', [x], [basis])

deg = 15
collocation_points,w = scipy.special.roots_legendre(deg)

p = Parameters(m_pend=0.15, l = 0.5, m_cart=0.1, g=9.8, nlinks=2)
d = Dynamics(p)

s = SX.sym('s')
basis_fun = get_lagrange_basis(collocation_points)
basis = basis_fun(s)
D_basis = jacobian(basis, s)

nq,_ = d.q.shape
n,_ = basis.shape
cq = SX.sym('cq', nq, n)
q = cq @ basis
dq = cq @ D_basis
L = d.K - d.U
L = substitute(L, vertcat(d.q, d.dq), vertcat(q, dq))
I = sum1([substitute(L, s, cp) for cp in collocation_points])
print(I)
