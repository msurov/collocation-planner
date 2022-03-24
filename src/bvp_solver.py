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

def solve_bvp(rhs_fun, bc_func, deg):
    nx = rhs_fun.numel_out()
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
        eq = dxcp - tspan * rhs_fun(xcp)
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


def solve_mechanical_bvp(sys, qleft, qright, deg):
    M = sys['M']
    C = sys['C']
    G = sys['G']
    B = sys['B']
    q = sys['q']
    dq = sys['dq']
    u = sys['u']
    s = SX.sym('s')

    nq,_ = q.shape
    nu,_ = u.shape

    # construct basis functions
    basis = find_orthogonal_basis(deg)
    D_basis = [b.diff() for b in basis]
    D2_basis = [b.diff() for b in D_basis]

    basis_coefs = [evalf_coefs(b) for b in basis]
    D_basis_coefs = [evalf_coefs(b) for b in D_basis]
    D2_basis_coefs = [evalf_coefs(b) for b in D2_basis]

    basis_polys = vertcat(*[polyval(c, s) for c in basis_coefs])
    D_basis_polys = vertcat(*[polyval(c, s) for c in D_basis_coefs])
    D2_basis_polys = vertcat(*[polyval(c, s) for c in D2_basis_coefs])

    basis_polys_left = vertcat(*[polyval(c, 0) for c in basis_coefs])
    basis_polys_right = vertcat(*[polyval(c, 1) for c in basis_coefs])

    D_basis_polys_left = vertcat(*[polyval(c, 0) for c in D_basis_coefs])
    D_basis_polys_right = vertcat(*[polyval(c, 1) for c in D_basis_coefs])

    # find collocation points
    bn = basis_coefs[-1]
    roots = np.roots(bn)
    collocation_points = np.sort(roots)        

    # q,dq,ddq,u decomposition
    nc = deg + 1
    cq = SX.sym('cq', nq, nc)
    q = cq @ basis_polys
    dq = cq @ D_basis_polys
    ddq = cq @ D2_basis_polys

    cu = SX.sym('cu', nu, nc)
    u = cu @ basis_polys

    # constraints
    constraints = []
    constraints_lb = []
    constraints_ub = []

    # equations
    T = SX.sym('T')
    constraints += [T]
    constraints_lb += [1e-2]
    constraints_ub += [1e+2]

    for cp in collocation_points:
        qcp = substitute(q, s, cp)
        dqcp = substitute(dq, s, cp)
        ddqcp = substitute(ddq, s, cp)
        ucp = substitute(u, s, cp)

        eq = M(qcp) @ ddqcp + C(qcp, dqcp) @ dqcp + G(qcp) * T**2 - B(qcp) @ ucp * T**2
        constraints += [eq]
        constraints_lb += [-1e-5 * DM.ones(eq.shape)]
        constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # boundary conditions
    eq = cq @ basis_polys_left - qleft
    constraints += [eq]
    constraints_lb += [-1e-5 * DM.ones(eq.shape)]
    constraints_ub += [1e-5 * DM.ones(eq.shape)]

    eq = cq @ D_basis_polys_left
    constraints += [eq]
    constraints_lb += [-1e-5 * DM.ones(eq.shape)]
    constraints_ub += [1e-5 * DM.ones(eq.shape)]

    eq = cq @ basis_polys_right - qright
    constraints += [eq]
    constraints_lb += [-1e-5 * DM.ones(eq.shape)]
    constraints_ub += [1e-5 * DM.ones(eq.shape)]

    eq = cq @ D_basis_polys_right
    constraints += [eq]
    constraints_lb += [-1e-5 * DM.ones(eq.shape)]
    constraints_ub += [1e-5 * DM.ones(eq.shape)]

    # cost function
    cost_function = T

    # control constraints
    for cp in collocation_points:
        ucp = substitute(u, s, cp)
        constraints += [ucp]
        constraints_lb += [-10]
        constraints_ub += [10]

    ucp = substitute(u, s, 0)
    constraints += [ucp]
    constraints_lb += [-10]
    constraints_ub += [10]

    ucp = substitute(u, s, 1)
    constraints += [ucp]
    constraints_lb += [-10]
    constraints_ub += [10]

    # compose NLP
    decision_variables = [reshape(cq, -1, 1)]
    decision_variables += [reshape(cu, -1, 1)]
    decision_variables += [T]

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

    B = basis_mat(basis_coefs)

    T_found = float(substitute(T, decision_variables, sol['x']))
    cq_found = DM(substitute(cq, decision_variables, sol['x']))
    q_poly_found = np.array(cq_found @ B)
    cu_found = DM(substitute(cu, decision_variables, sol['x']))
    u_poly_found = np.array(cu_found @ B)

    return q_poly_found.T, u_poly_found.T, T_found


def solve_control(rhs_fun, bc_left_fun, bc_right_fun, deg):
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

    # find collocation points
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
        eq = dxcp - tspan * rhs_fun(xcp, ucp)
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

    # cost function
    cost_function = tspan
    # for cp in collocation_points:
    #     ucp = substitute(u, s, cp)
    #     cost_function += ucp.T @ ucp

    # control constraints
    for cp in collocation_points:
        ucp = substitute(u, s, cp)
        constraints += [ucp]
        constraints_lb += [-10]
        constraints_ub += [10]

    # compose NLP
    decision_variables = [reshape(cx, -1, 1)]
    decision_variables += [reshape(cu, -1, 1)]
    decision_variables += [tspan]

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

    B = basis_mat(basis_coefs)

    tspan_found = float(substitute(tspan, decision_variables, sol['x']))
    cx_found = DM(substitute(cx, decision_variables, sol['x']))
    x_poly_found = np.array(cx_found @ B)
    cu_found = DM(substitute(cu, decision_variables, sol['x']))
    u_poly_found = np.array(cu_found @ B)

    return x_poly_found.T, u_poly_found.T, tspan_found


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
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=200)
    test_solve_bvp()
