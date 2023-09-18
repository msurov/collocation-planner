from casadi import sin, cos, pi, SX, DM, \
    jtimes, jacobian, vertcat, horzcat, \
    substitute, Function, reshape, nlpsol, power, solve, polyval, pinv, DM_inf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cartpend_dynamics import Parameters, Dynamics
from cartpend_anim import CartPendAnim
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

def get_cgl_collocation_points(N):
    i = np.arange(N)
    cp = -np.cos(np.pi * i / (N - 1))
    return cp

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

def solve_mechanical_opt(sys, ql, qr, umin, umax, deg, eps=1e-7):
    q = sys['q']
    dq = sys['dq']
    M = Function('M', [q], [sys['M']])
    C = Function('C', [q,dq], [sys['C']])
    B = Function('B', [q], [sys['B']])
    G = Function('G', [q], [sys['G']])
    Bperp = Function('Bperp', [q], [sys['Bperp']])
    nq,_ = q.shape

    s = SX.sym('s')
    collocation_points = get_lgl_collocation_points(deg + 1)
    # collocation_points = get_uniform_collocation_points(deg + 1)
    basis_fun = get_lagrange_basis(collocation_points)
    basis = basis_fun(s)
    n,_ = basis.shape
    D_basis = jacobian(basis, s)
    D2_basis = jacobian(D_basis, s)
    s1, s2 = -1, 1
    basis_left = substitute(basis, s, s1)
    basis_right = substitute(basis, s, s2)
    D_basis_left = substitute(D_basis, s, s1)
    D_basis_right = substitute(D_basis, s, s2)

    cq = SX.sym('cq', nq, n)
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
    Tinv_sq = SX.sym('Tinv_sq')
    constraints += [Tinv_sq]
    constraints_lb += [1e-2]
    constraints_ub += [1e+2]

    k = Tinv_sq * (s2 - s1)**2
    sys = Bperp(q) @ (
        k * (M(q) @ d2q + C(q,dq) @ dq) + G(q)
    )

    for cp in collocation_points:
        eq = substitute(sys, s, cp)
        constraints += [eq]
        constraints_lb += [-eps * DM.ones(eq.shape)]
        constraints_ub += [eps * DM.ones(eq.shape)]

    # control constraints
    Binv = pinv(B(q))
    k = (s2 - s1)**2 * Tinv_sq
    u_expr = Binv @ (M(q) @ d2q * k + C(q,dq) @ dq * k + G(q))

    for cp in collocation_points:
        ucp = substitute(u_expr, s, cp)
        constraints += [ucp]
        constraints_lb += [umin * DM.ones(ucp.shape)]
        constraints_ub += [umax * DM.ones(ucp.shape)]

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

    # generalized velocities
    for cp in collocation_points:
        eq = substitute(dq, s, cp)
        constraints += [eq]
        constraints_lb += [-10 * DM.ones(eq.shape)]
        constraints_ub += [10 * DM.ones(eq.shape)]

    # compose NLP
    decision_variables = [reshape(cq, -1, 1)]
    decision_variables += [Tinv_sq]
    cost_function = -Tinv_sq

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
    dv0 = substitute(decision_variables, cq, 1e-2*np.random.normal(size=cq.shape))
    dv0 = substitute(dv0, Tinv_sq, 1)
    dv0 = DM(dv0)

    # solve NLP
    BVP = nlpsol('BVP', 'ipopt', nlp)
    sol = BVP(x0=dv0, lbg=constraints_lb, ubg=constraints_ub)
    stat = BVP.stats()
    if not stat['success']:
        return None

    q_found = substitute(q, decision_variables, sol['x'])
    q_func = Function('q', [s], [q_found])
    Tinv_sq_found = float(substitute(Tinv_sq, decision_variables, sol['x']))
    T_found = 1 / np.sqrt(Tinv_sq_found)
    u_found = substitute(u_expr, decision_variables, sol['x'])
    u_func = Function('u', [s], [u_found])
    dq_found = substitute(dq, decision_variables, sol['x'])
    dq_func = Function('dq', [s], [dq_found * (s2 - s1) / T_found])

    ss = np.linspace(s1, s2, 1000)
    tt = (ss - s1) * T_found / (s2 - s1)
    qq = np.array([q_func(si) for si in ss], float)[:,:,0]
    dqq = np.array([dq_func(si) for si in ss], float)[:,:,0]
    uu = np.array([u_func(si) for si in ss], float)[:,:,0]

    return tt, qq, dqq, uu


def test_solve_mech_opt():
    p = Parameters(m_pend=0.15, l = 0.5, m_cart=0.1, g=9.8, nlinks=2)
    d = Dynamics(p)
    sys = {
        'M': d.M,
        'C': d.C,
        'G': d.G,
        'B': d.B,
        'Bperp': d.Bperp,
        'q': d.q,
        'dq': d.dq,
        'u': d.u
    }
    ql = DM([0] + [pi] * p.nlinks)
    qr = DM([-1] + [0] * p.nlinks)

    args = (sys, ql, qr, -50, 50, 17)
    ans = None
    while ans is None:
        ans = solve_mechanical_opt(*args)

    t, q, dq, u = ans
    simdata = {
        't': t,
        'q': q,
        'u': u
    }

    M_fun = Function('M', [d.q], [d.M])
    U_fun = Function('U', [d.q], [d.U])
    B = np.array(DM(d.B), float)
    E = np.zeros(len(t))
    W = np.zeros(len(t))

    for i in range(len(t)):
        qi = q[i]
        dqi = dq[i]
        ui = u[i]
        M = np.array(M_fun(qi), float)
        U = float(U_fun(qi))
        W[i] = dqi.T @ B @ ui
        E[i] = dqi.T @ M @ dqi / 2 + U

    plt.figure('power')
    plt.plot(t, W)
    plt.plot(t[1:], np.diff(E) / np.diff(t))

    plt.figure('u(t)')
    plt.plot(t, u)

    plt.figure('q(t)')
    plt.plot(t, q)
    plt.show()

    anim = CartPendAnim('fig/cartpend.svg', nlinks=p.nlinks)
    anim.run(simdata, filepath='data/anim.gif', animtime=5)


if __name__ == '__main__':
    test_solve_mech_opt()
