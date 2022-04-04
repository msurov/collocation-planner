from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv, DM_inf, \
    diagcat, repmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cartpend_dynamics import Parameters, Dynamics
from cartpend_anim import CartPendAnim
from basis import get_basis, get_collocation_points, \
    get_diap, get_legendre_roots, get_cheb1_roots, get_cheb2_roots


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


def solve_bvp(rhs_func, bc_func, T, deg):
    nx,_ = rhs_func.size_in(0)
    s = SX.sym('s')

    s1,s2 = -1,1
    n = deg
    collocation_points = get_legendre_roots(deg)
    # collocation_points = get_cheb1_roots(deg)
    basis_fun = get_lagrange_basis(collocation_points)
    basis = basis_fun(s)
    D_basis = jacobian(basis, s)

    cx = SX.sym('cx', nx, n)
    x = cx @ basis
    dx = cx @ D_basis

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # differential equations
    for cp in collocation_points:
        xcp = substitute(x, s, cp)
        dxcp = substitute(dx, s, cp)
        eq = rhs_func(xcp) * T / (s2 - s1) - dxcp

        constraints += [eq]
        constraints_lb += [-1e-5 * DM.ones(eq.shape)]
        constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # boundary conditions
    xleft = substitute(x, s, collocation_points[0])
    xright = substitute(x, s, collocation_points[-1])
    eq = bc_func(xleft, xright)
    constraints += [eq]
    constraints_lb += [-1e-5 * DM.ones(eq.shape)]
    constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # compose NLP
    decision_variables = [reshape(cx, -1, 1)]
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
    dv0 = substitute(decision_variables, cx, DM.zeros(cx.shape))
    dv0 = DM(dv0)

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)

    F = Function('F', [s], [substitute(x, decision_variables, sol['x'])])
    ss = np.linspace(s1, s2, 100)
    tt = (ss - s1) * T / (s2 - s1)
    yy = np.array([F(si) for si in ss])[:,:,0]
    plt.plot(tt, yy[:,0])
    plt.show()


def test1():
    x = SX.sym('x', 2)
    mu = 1.
    rhs = Function('RHS', [x], [
        vertcat(
            x[1],
            mu * (1 - x[0]**2) * x[1] - x[0]
        )
    ])

    xl = SX.sym('xl', x.shape)
    xr = SX.sym('xr', x.shape)
    bc_func = Function('BC', [xl, xr], [vertcat(xl[0] - 0, xr[0] - 1)])
    solve_bvp(rhs, bc_func, 5, 15)


if __name__ == '__main__':
    test1()
