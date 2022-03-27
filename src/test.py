from pydoc import describe
from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sym_poly import find_orthogonal_basis
import scipy.special
from cartpend_dynamics import Parameters, Dynamics
from cartpend_anim import CartPendAnim


def get_cheb_basis(deg, kind=1):
    assert kind in [1,2]
    x = SX.sym('x')
    b0 = 1
    b1 = kind * x
    basis = [b0, b1]
    for k in range(2, deg + 1):
        bk = 2*x*basis[-1] - basis[-2]
        basis.append(bk)
    basis = vertcat(*basis)
    basis_fun = Function(f'Cheb{kind}', [x], [basis])
    return basis_fun

def get_diap(basis_fun):
    name = basis_fun.name()
    if name == 'Legendre':
        return -1.,1.
    elif name == 'Cheb1':
        return -1.,1.
    elif name == 'Cheb2':
        return -1.,1.
    else:
        assert False

def get_legendre_basis(deg):
    x = SX.sym('x')
    basis = []
    for k in range(deg + 1):
        bk = scipy.special.legendre(k)
        poly = polyval(bk.coefficients, x)
        basis += [poly]
    basis = vertcat(*basis)
    basis_fun = Function('Legendre', [x], [basis])
    return basis_fun


def get_basis(deg):
    return get_cheb_basis(deg, kind=1)
    # return get_legendre_basis(deg)


def get_collocation_points(basis_fun):
    name = basis_fun.name()
    deg = basis_fun.numel_out() - 1
    if name == 'Legendre':
        r,_ = scipy.special.roots_legendre(deg)
    elif name == 'Cheb1':
        r,_ = scipy.special.roots_chebyt(deg)
    elif name == 'Cheb2':
        r,_ = scipy.special.roots_chebyu(deg)
    else:
        assert False
    return r


def solve_bvp(rhs_func, bc_func, deg):
    nx,_ = rhs_func.size_in(0)
    s = SX.sym('s')
    T = SX.sym('T')

    basis_fun = get_basis(deg)
    basis = basis_fun(s)
    n,_ = basis.shape
    D_basis = jacobian(basis, s)
    s1, s2 = get_diap()
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


def solve_mech_ivp(sys, q0, dq0, T, uoft):
    q = sys['q']
    dq = sys['dq']
    M = Function('M', [q], [sys['M']])
    C = Function('C', [q,dq], [sys['C']])
    G = Function('G', [q], [sys['G']])
    B = Function('B', [q], [sys['B']])
    u = sys['u']
    ddq = pinv(M(q)) @ (-C(q,dq) @ dq - G(q) + B(q) @ u)
    F = Function('F', [vertcat(q, dq), u], [vertcat(dq, ddq)])

    def rhs(t, st):
        u = uoft(t)
        dst = F(st, u)
        return np.reshape(dst, -1)

    st0 = np.reshape(vertcat(q0, dq0), -1)
    sol = solve_ivp(rhs, [0,T], st0, max_step=T / 1000)
    return sol


def solve_mechanical_bvp(sys, ql, qr, umin, umax, deg, eps=1e-7):
    q = sys['q']
    u = sys['u']
    dq = sys['dq']
    M = Function('M', [q], [sys['M']])
    C = Function('C', [q,dq], [sys['C']])
    G = Function('G', [q], [sys['G']])
    B = Function('B', [q], [sys['B']])
    nq,_ = q.shape
    nu,_ = u.shape

    s = SX.sym('s')
    T = SX.sym('T')
    basis_fun = get_basis(deg)
    basis = basis_fun(s)
    n,_ = basis.shape
    D_basis = jacobian(basis, s)
    D2_basis = jacobian(D_basis, s)
    s1, s2 = get_diap(basis_fun)
    collocation_points = get_collocation_points(basis_fun)
    basis_left = substitute(basis, s, s1)
    basis_right = substitute(basis, s, s2)
    D_basis_left = substitute(D_basis, s, s1)
    D_basis_right = substitute(D_basis, s, s2)

    cq = SX.sym('cq', nq, n)
    cu = SX.sym('cu', nu, n)
    u = cu @ basis
    q = cq @ basis
    dq = cq @ D_basis
    d2q = cq @ D2_basis
    q_left = cq @ basis_left
    q_right = cq @ basis_right
    dq_left = cq @ D_basis_left
    dq_right = cq @ D_basis_right

    constraints = []
    constraints_lb = []
    constraints_ub = []

    # differential equations
    constraints += [T]
    constraints_lb += [1e-2]
    constraints_ub += [1e+2]

    eq = M(q) @ d2q + C(q,dq) @ dq + (G(q) - B(q) @ u) * T**2 / (s2 - s1)**2

    for cp in collocation_points:
        tmp = substitute(eq, s, cp)
        constraints += [tmp]
        constraints_lb += [-eps * DM.ones(tmp.shape)]
        constraints_ub += [eps * DM.ones(tmp.shape)]

    # boundary conditions
    eq = q_left - ql
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = q_right - qr
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = dq_left
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    eq = dq_right
    constraints += [eq]
    constraints_lb += [-eps * DM.ones(eq.shape)]
    constraints_ub += [eps * DM.ones(eq.shape)]

    # control constraints
    for cp in collocation_points:
        tmp = substitute(u, s, cp)
        constraints += [tmp]
        constraints_lb += [umin * DM.ones(tmp.shape)]
        constraints_ub += [umax * DM.ones(tmp.shape)]

    tmp = substitute(u, s, s1)
    constraints += [tmp]
    constraints_lb += [umin * DM.ones(tmp.shape)]
    constraints_ub += [umax * DM.ones(tmp.shape)]

    tmp = substitute(u, s, s2)
    constraints += [tmp]
    constraints_lb += [umin * DM.ones(tmp.shape)]
    constraints_ub += [umax * DM.ones(tmp.shape)]

    # compose NLP
    decision_variables = [reshape(cq, -1, 1)]
    decision_variables += [reshape(cu, -1, 1)]
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
    dv0 = substitute(dv0, cq, 1e-2*np.random.normal(size=cq.shape))
    dv0 = substitute(dv0, cu, 1e-2*np.random.normal(size=cu.shape))
    dv0 = DM(dv0)

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)
    values = DM(substitute(constraints, decision_variables, sol['x']))
    stat = BVP.stats()
    if not stat['success']:
        return None

    q_found = substitute(q, decision_variables, sol['x'])
    u_found = substitute(u, decision_variables, sol['x'])
    dq_found = substitute(dq, decision_variables, sol['x'])
    d2q_found = substitute(d2q, decision_variables, sol['x'])
    q_func = Function('Q', [s], [q_found])
    dq_func = Function('DQ', [s], [dq_found])
    T_found = float(substitute(T, decision_variables, sol['x']))
    
    ss = np.linspace(s1, s2, 100)
    tt = np.array((ss - s1) / (s2 - s1) * T_found, float)
    qq = qq = np.array([q_func(si) for si in ss], float)[:,:,0]
    return tt, qq


def evalf_coefs(poly):
    return np.array(poly.all_coeffs(), float)


def fit(x, y, N):
    s = SX.sym('s')
    k = np.arange(1, N + 1)
    xmin = np.min(x)
    xmax = np.max(x)

    basis_fun = get_cheb_basis(N, kind=2)
    basis = basis_fun(s)
    ss = 2 * (x - xmin) / (xmax - xmin) - 1

    n,_ = basis.shape
    basis_fun = Function('basis', [s], [basis])

    A = DM.zeros(n, n)
    B = DM.zeros(n, 1)
    for si,yi in zip(ss, y):
        b = basis_fun(si)
        A += b @ b.T
        B += b @ DM(yi)

    coefs = solve(A, B)
    y_fun = Function('y', [s], [coefs.T @ basis])
    yy = np.array([y_fun(si) for si in ss])[:,:,0]

    plt.plot(x, y, 'o')
    plt.plot(x, yy)
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


def test_solve_mech_bvp_1():
    p = Parameters(m_pend=0.5, l = 0.5, m_cart=0.1, g=9.8, nlinks=1)
    d = Dynamics(p)
    sys = {
        'M': d.M,
        'C': d.C,
        'G': d.G,
        'B': d.B,
        'q': d.q,
        'dq': d.dq,
        'u': d.u
    }
    ql = DM([0, pi])
    qr = DM([0, 0])
    ans = solve_mechanical_bvp(sys, ql, qr, -50, 50, 7)
    simdata = {
        't': t,
        'q': q
    }
    anim = CartPendAnim('fig/cartpend.svg')
    anim.run(simdata, animtime=5, filepath='data/anim.mp4')


def test_solve_mech_bvp_2():
    p = Parameters(m_pend=0.5, l = 0.5, m_cart=0.1, g=9.8, nlinks=3)
    d = Dynamics(p)
    sys = {
        'M': d.M,
        'C': d.C,
        'G': d.G,
        'B': d.B,
        'q': d.q,
        'dq': d.dq,
        'u': d.u
    }
    ql = DM([0, pi, pi, pi])
    qr = DM([0, 0, 0, 0])
    ans = None
    while ans is None:
        ans = solve_mechanical_bvp(sys, ql, qr, -20, 20, 15)

    t, q = ans
    simdata = {
        't': t,
        'q': q
    }
    anim = CartPendAnim('fig/cartpend.svg', nlinks=p.nlinks)
    anim.run(simdata, filepath='data/anim.mp4')


def test_basis_roots():
    basis_fun = get_legendre_basis(15)
    # basis_fun = get_legendre_basis(11, kind=1)
    # basis_fun = get_legendre_basis(11, kind=2)
    s1,s2 = get_diap(basis_fun)
    cp = get_collocation_points(basis_fun)
    ss = np.linspace(s1, s2, 100)
    bb = np.array([basis_fun(si) for si in ss])[:,-1,0]
    plt.plot(ss, bb)
    plt.plot(cp, cp*0, 'o')
    plt.show()
    

# test_solve_mech_bvp_1()
test_solve_mech_bvp_2()
