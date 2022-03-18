import numpy as np
from casadi import MX,SX,DM,polyval,vertcat,substitute,jacobian,Function,reshape,horzcat,nlpsol,sin
from poly import orthonormal_basis,decompose,basis_mat_form
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


deg = 11
nc = deg + 1
nx = 2
ncolpts = deg
basis = orthonormal_basis(deg)
roots = np.roots(basis[-1])
colpts = np.sort(roots)
c = SX.sym('c', nx, nc)
t = SX.sym('t')

x00 = 3
x01 = 2

basis_funcs = vertcat(*[polyval(b, t) for b in basis])
basis_funcs_at_left = vertcat(*[polyval(b, -1) for b in basis])
basis_funcs_at_right = vertcat(*[polyval(b, 1) for b in basis])

xx = SX.sym('x', 2)
F = Function('RHS', [xx], [vertcat(xx[1], xx[1]**2 * sin(xx[0]) - xx[0])])

x = c @ basis_funcs
x_left = c @ basis_funcs_at_left
x_right = c @ basis_funcs_at_right
x_at_colpts = [substitute(x, t, cp) for cp in colpts]
dx = jacobian(x, t)
dx_at_colpts = [substitute(dx, t, cp) for cp in colpts]
F_at_colpts = [F(w) for w in x_at_colpts]

eqs = [(a - b) for a,b in zip(dx_at_colpts, F_at_colpts)]
eqs = vertcat(*eqs)
eqs = vertcat(eqs, x_left[0] - x00, x_right[0] - x01)

c_flat = reshape(c, -1, 1)
nlp = {
    'x': c_flat,
    'f': 1,
    'g': eqs
}

BVP = nlpsol('B', 'ipopt', nlp)
sol = BVP(lbg=-1e-5,ubg=1e-5)
x_found = substitute(x, c_flat, sol['x'])
x_fun = Function('x', [t], [x_found])

tt = np.linspace(-1, 1)
xx = np.array([x_fun(w)[:,0] for w in tt], float)
xx = xx[:,:,0]

plt.plot(tt, xx[:,0])
plt.plot(tt, xx[:,1])
plt.plot(-1, x00, 'o')
plt.plot(1, x01, 'o')

def rhs(_,x):
    dx = F(x)
    return np.reshape(dx, -1)

x0 = np.array(xx[0,:], float)
tspan = [-1, 1]
sol = solve_ivp(rhs, tspan, x0, max_step=1e-3)
plt.plot(sol.t, sol.y[0], '--')
plt.plot(sol.t, sol.y[1], '--')

plt.show()
