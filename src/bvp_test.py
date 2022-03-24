import numpy as np
from casadi import MX,SX,DM,polyval,vertcat,substitute,\
    Function,reshape,nlpsol,sin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sym_poly import find_orthonormal_basis, find_orthogonal_basis


def evalf_coefs(poly):
    return np.array(poly.all_coeffs(), float)

def basis_mat(basis):
    n = len(basis)
    B = np.zeros((n, n))
    for i,b in enumerate(basis):
        B[i,n-i-1:] = b
    return B

def poly_scale_arg(poly, scale):
    poly = np.copy(poly)
    n = np.shape(poly)[-1]
    for i in range(n):
        poly[...,i] *= np.power(scale, i)
    return poly

def solve_bvp(rhs_func, bc_func, deg):
    nx = rhs_func.numel_out()
    s = SX.sym('s')

    # construct basis functions
    basis = find_orthogonal_basis(deg)
    basis_coefs = [evalf_coefs(b) for b in basis]
    deriv_basis_coefs = [evalf_coefs(b.diff()) for b in basis]
    basis_polys = vertcat(*[polyval(c, s) for c in basis_coefs])
    basis_polys_left = vertcat(*[polyval(c, 0) for c in basis_coefs])
    basis_polys_right = vertcat(*[polyval(c, 1) for c in basis_coefs])
    deriv_basis_polys = vertcat(*[polyval(c, s) for c in deriv_basis_coefs])

    bn = basis_coefs[-1]
    roots = np.roots(bn)
    collocation_points = np.sort(roots)        

    nc = deg + 1
    cx = SX.sym('cx', nx, nc)
    x = cx @ basis_polys
    dx = cx @ deriv_basis_polys

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # differential equation at collocation points
    tspan = SX.sym('T')
    constraints += [tspan]
    constraints_lb += [1e-2]
    constraints_ub += [1e+2]

    for cp in collocation_points:
        xcp = substitute(x, s, cp)
        dxcp = substitute(dx, s, cp)
        eq = dxcp - tspan * rhs_func(xcp)
        constraints += [eq]
        constraints_lb += [-1e-5 * DM.ones(eq.shape)]
        constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # boundary conditions
    xl = cx @ basis_polys_left
    xr = cx @ basis_polys_right

    tmp = bc_func(xl, xr)
    constraints += [tmp]
    constraints_lb += [-1e-5 * DM.ones(tmp.shape)]
    constraints_ub += [1e-5 * DM.ones(tmp.shape)]

    # compose NLP
    decision_variables = [reshape(cx, -1, 1)]
    decision_variables += [tspan]
    cost_function = tspan

    decision_variables = vertcat(*decision_variables)
    constraints = vertcat(*constraints)
    constraints_lb = vertcat(*constraints_lb)
    constraints_ub = vertcat(*constraints_ub)

    nlp = {
        'x': decision_variables,
        'f': cost_function,
        'g': constraints
    }

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    dv0 = np.zeros(decision_variables.shape)
    dv0[0] = 1
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)

    cx_found = DM(substitute(cx, decision_variables, sol['x']))
    tspan_found = float(substitute(tspan, decision_variables, sol['x']))
    x_poly_found = np.array(cx_found @ basis_mat(basis_coefs))

    return x_poly_found.T, tspan_found

def solve_control(rhs_fun, bc_left_fun, bc_rigt_fun, deg):
    nx,_ = rhs_fun.size_in(0)
    nu,_ = rhs_fun.size_in(1)
    s = SX.sym('s')

    # construct basis functions
    basis = find_orthogonal_basis(deg)
    basis_coefs = [evalf_coefs(b) for b in basis]
    deriv_basis_coefs = [evalf_coefs(b.diff()) for b in basis]
    basis_polys = vertcat(*[polyval(c, s) for c in basis_coefs])
    basis_polys_left = vertcat(*[polyval(c, 0) for c in basis_coefs])
    basis_polys_right = vertcat(*[polyval(c, 1) for c in basis_coefs])
    deriv_basis_polys = vertcat(*[polyval(c, s) for c in deriv_basis_coefs])

    bn = basis_coefs[-1]
    roots = np.roots(bn)
    collocation_points = np.sort(roots)        

    nc = deg + 1
    cx = SX.sym('cx', nx, nc)
    x = cx @ basis_polys
    dx = cx @ deriv_basis_polys

    cu = SX.sym('cu', nu, nc)
    u = cu @ basis_polys

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # differential equation at collocation points
    tspan = SX.sym('T')
    constraints += [tspan]
    constraints_lb += [1e-2]
    constraints_ub += [1e+2]

    for cp in collocation_points:
        xcp = substitute(x, s, cp)
        dxcp = substitute(dx, s, cp)
        ucp = substitute(u, s, cp)
        eq = dxcp - tspan * rhs_func(xcp, ucp)
        constraints += [eq]
        constraints_lb += [-1e-5 * DM.ones(eq.shape)]
        constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # boundary conditions
    xl = cx @ basis_polys_left
    tmp = bc_left_fun(xl)
    constraints += [tmp]
    constraints_lb += [-1e-5 * DM.ones(tmp.shape)]
    constraints_ub += [1e-5 * DM.ones(tmp.shape)]

    xr = cx @ basis_polys_right
    tmp = bc_right_fun(xr)
    constraints += [tmp]
    constraints_lb += [-1e-5 * DM.ones(tmp.shape)]
    constraints_ub += [1e-5 * DM.ones(tmp.shape)]

    # compose NLP
    decision_variables = [reshape(cx, -1, 1)]
    decision_variables += [reshape(cu, -1, 1)]
    decision_variables += [tspan]
    cost_function = tspan

    decision_variables = vertcat(*decision_variables)
    constraints = vertcat(*constraints)
    constraints_lb = vertcat(*constraints_lb)
    constraints_ub = vertcat(*constraints_ub)

    nlp = {
        'x': decision_variables,
        'f': cost_function,
        'g': constraints
    }

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    dv0 = np.zeros(decision_variables.shape)
    dv0[0] = 1
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)

    cx_found = DM(substitute(cx, decision_variables, sol['x']))
    tspan_found = float(substitute(tspan, decision_variables, sol['x']))
    x_poly_found = np.array(cx_found @ basis_mat(basis_coefs))

    return x_poly_found.T, tspan_found


def test_solve_bvp():
    x = SX.sym('x', 2)
    rhs = Function('RHS', [x], [vertcat(x[1], x[1]**2 * sin(x[0]) - x[0])])

    xl = SX.sym('x', 2)
    xr = SX.sym('x', 2)
    bc_func = Function('BC', [xl, xr], [vertcat(xl[0] - 2, xr[0] - 5)])

    x_poly,tspan = solve_bvp(rhs, bc_func, 15)

    sol = solve_ivp(
        lambda _,x: np.reshape(rhs(x), -1),
        [0, tspan],
        np.polyval(x_poly, 0),
        max_step=tspan*1e-2
    )

    s = np.linspace(0, 1, 100)
    x1 = np.polyval(x_poly[:,0], s)
    x2 = np.polyval(x_poly[:,1], s)
    plt.plot(s * tspan, x1)
    plt.plot(sol.t, sol.y[0])
    # plt.plot(s * tspan, x2)
    # plt.plot(sol.t, sol.y[1])
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=200)
    # test_solve_bvp()
    solve_control()
