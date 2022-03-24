import sympy as sy
import numpy as np
import matplotlib.pyplot as plt


def get_limits():
    return [0, 1]

def prod(f, g):
    I = (f * g).integrate()
    x1,x2 = get_limits()
    return I(x2) - I(x1)

def norm_sq(f):
    return prod(f, f)

def find_orthonormal_basis(deg):
    nelems = deg + 1
    assert nelems > 0
    domain = 'EX'

    x = sy.Dummy(real=True)
    b = sy.Poly(1, x, domain=domain)
    b,_ = b.div(sy.sqrt(norm_sq(b)))
    basis = [b]

    for i in range(1, nelems):
        q = sy.Poly(0, x, domain=domain)
        p = sy.Poly(x**i, x, domain=domain)
        for j in range(i):
            q += prod(p, basis[j]) * basis[j]
        b = p - q
        b,_ = b.div(sy.sqrt(norm_sq(b)))
        basis += [b]
    return basis

def find_orthogonal_basis(deg):
    nelems = deg + 1
    assert nelems > 0
    domain = 'EX'

    x = sy.Dummy(real=True)
    b = sy.Poly(1, x, domain=domain)
    b,_ = b.div(sy.sqrt(norm_sq(b)))
    basis = [b]

    for i in range(1, nelems):
        q = sy.Poly(0, x, domain=domain)
        p = sy.Poly(x**i, x, domain=domain)
        for j in range(i):
            q += prod(p, basis[j]) * basis[j] / prod(basis[j], basis[j])
        b = p - q
        basis += [b]
    return basis

def test_orthonormal_basis():
    basis = find_orthonormal_basis(10)

    for i,b in enumerate(basis):
        print(f'b{i} = {b}')

    for i in range(1, len(basis)):
        for j in range(len(basis)):
            d = prod(basis[i], basis[j])
            assert d == ((i == j) * 1)

def plot_basis():
    basis = find_orthonormal_basis(10)
    basis_funcs = []
    for b in basis:
        expr,arg = b.args
        basis_funcs += [sy.lambdify(arg, expr)]

    x1, x2 = get_limits()
    x = np.linspace(x1, x2, 1000)

    y = np.zeros((len(x), len(basis)))
    for i,f in enumerate(basis_funcs):
        y[:,i] = f(x)

    plt.plot(x, y)
    plt.grid()
    plt.show()

def decompose(f, basis):
    return [prod(f, b) for b in basis]

def decompose_deriv(k):
    '''
        decompose d bk(x) / dx
    '''
    coefs = sy.zeros(k, 1)
    n0 = (k + 1) % 2
    for n in range(n0, k, 2):
        v = (2*k + 1) * (2*n + 1)
        coefs[n] = [2 * sy.sqrt(v)]
    return coefs

def test_decompose_derivs():
    for k in range(8, 15):
        basis = find_orthonormal_basis(k)
        bk = basis[k]
        Dbk = bk.diff()
        coefs = decompose(Dbk, basis[:-1])
        coefs = sy.Matrix(coefs)
        coefs2 = decompose_deriv(k)
        assert coefs == coefs2

def test_orthogonal_basis():
    deg = 11
    basis = find_orthogonal_basis(deg)
    nelems = len(basis)
    assert nelems == deg + 1

    for i in range(nelems):
        for j in range(nelems):
            s = prod(basis[i], basis[j])
            assert s == 0 or i == j

if __name__ == '__main__':
    test_decompose_derivs()
    test_orthonormal_basis()
    plot_basis()
    test_orthogonal_basis()
