from pydoc import describe
from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv, DM_inf, \
    diagcat, repmat, poly_roots
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import jacobi
from cartpend_dynamics import Parameters, Dynamics
from cartpend_anim import CartPendAnim
from basis import get_basis, get_collocation_points, \
    get_diap, get_legendre_roots, get_cheb1_roots, get_cheb2_roots
import scipy.special


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

def get_lgl_collocation_points(N):
    R'''
        Legendre-Gauss-Lobatto collocation points
        are the roots of the poly dL_[N-1] / dx
        degree of the poly is (N - 2)
        there are (N - 2) roots 
        The collocation points also include {-1,1}

        `N` is the number of collocation points to retrieve
    '''
    bk = scipy.special.legendre(N - 1)
    coefs = np.polyder(bk.coefficients)
    r = np.roots(coefs)
    r = np.sort(r)
    return np.concatenate(([-1], r, [1]))

def get_uniform_collocation_points(N):
    return np.linspace(-1, 1, N)

def solve_bvp(rhs_func, bc_func, T, deg, eps=1e-7):
    nx,_ = rhs_func.size_in(0)
    s = SX.sym('s')

    s1,s2 = -1,1
    collocation_points = get_lgl_collocation_points(deg + 1)
    basis_fun = get_lagrange_basis(collocation_points)
    basis = basis_fun(s)
    D_basis = jacobian(basis, s)

    n,_ = basis.shape
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
        constraints_lb += [-eps * DM.ones(eq.shape)]
        constraints_ub += [eps * DM.ones(eq.shape)]

    # boundary conditions
    xleft = substitute(x, s, collocation_points[0])
    xright = substitute(x, s, collocation_points[-1])
    eq = bc_func(xleft, xright)
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

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

    tmp = substitute(constraints, decision_variables, sol['x'])
    tmp = np.array(DM(tmp).T, float)
    print('constr', tmp)

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
    bc_func = Function('BC', [xl, xr], [vertcat(xl[0] - 0, xr[0] - 3)])
    solve_bvp(rhs, bc_func, 0.5, 19)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    test1()
