from casadi import SX, DM, vertcat, Function, polyval
import scipy.special
import numpy as np


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

def get_basis(deg, name='Legendre'):
    if name == 'Legendre':
        return get_legendre_basis(deg)
    elif name == 'Cheb1':
        return get_cheb_basis(deg, kind=1)
    elif name == 'Cheb2':
        return get_cheb_basis(deg, kind=2)
    else:
        assert False


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


def test_basis_roots():
    import matplotlib.pyplot as plt

    basis_fun = get_basis(15, 'Cheb1')
    s1,s2 = get_diap(basis_fun)
    cp = get_collocation_points(basis_fun)
    ss = np.linspace(s1, s2, 1000)
    bb = np.array([basis_fun(si) for si in ss])[:,-1,0]
    plt.plot(ss, bb)
    plt.plot(cp, cp*0, 'o')
    plt.show()


if __name__ == '__main__':
    test_basis_roots()
