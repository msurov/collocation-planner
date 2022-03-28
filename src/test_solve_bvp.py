from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv, DM_inf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cartpend_dynamics import Parameters, Dynamics
from cartpend_anim import CartPendAnim
from basis import get_basis, get_collocation_points, get_diap


def solve_bvp(rhs_func, bc_func, deg):
    nx,_ = rhs_func.size_in(0)
    s = SX.sym('s')
    T = SX.sym('T')

    basis_fun = get_basis(deg, 'Legendre')
    basis = basis_fun(s)
    n,_ = basis.shape
    D_basis = jacobian(basis, s)
    s1, s2 = get_diap(basis_fun)
    collocation_points = get_collocation_points(basis_fun)

    basis_left = substitute(basis, s, s1)
    basis_right = substitute(basis, s, s2)

    cx = SX.sym('cx', nx, n)
    x = cx @ basis
    dx = cx @ D_basis
    x_left = cx @ basis_left
    x_right = cx @ basis_right

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # differential equations
    constraints += [T]
    constraints_lb += [1e-2]
    constraints_ub += [1e+2]

    for cp in collocation_points:
        xcp = substitute(x, s, cp)
        dxcp = substitute(dx, s, cp)
        eq = rhs_func(xcp) * T - dxcp

        constraints += [eq]
        constraints_lb += [-1e-5 * DM.ones(eq.shape)]
        constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # boundary conditions
    eq = bc_func(x_left, x_right)
    constraints += [eq]
    constraints_lb += [-1e-5 * DM.ones(eq.shape)]
    constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # compose NLP
    decision_variables = [reshape(cx, -1, 1)]
    decision_variables += [T]
    cost_function = T

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
    dv0 = substitute(dv0, cx, DM.zeros(cx.shape))
    dv0 = DM(dv0)

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)
    F = Function('F', [s], [substitute(x, decision_variables, sol['x'])])
    ss = np.linspace(s1, s2, 100)
    yy = np.array([F(si) for si in ss])[:,:,0]
    plt.plot(yy[:,0], yy[:,1])
    plt.show()


def test1():
    x = SX.sym('x', 3)
    beta = 8/3
    rho = 28
    sigma = 10
    rhs = Function('RHS', [x], [
        vertcat(
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        )
    ])

    xl = SX.sym('xl', x.shape)
    xr = SX.sym('xr', x.shape)
    bc_func = Function('BC', [xl, xr], [vertcat(xl[0] - 0, xr[0] - 1, xl[1] - 3)])
    solve_bvp(rhs, bc_func, 11)


if __name__ == '__main__':
    test1()
