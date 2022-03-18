import sympy as sy
import numpy as np
import matplotlib.pyplot as plt


def get_limits():
    return [-1, 1]

def prod(f, g):
    I = (f * g).integrate()
    x1,x2 = get_limits()
    return I(x2) - I(x1)

def norm_sq(f):
    return prod(f, f)

def find_orthonormal_basis(arg, nelems):
    assert nelems > 0
    x = arg
    domain = 'EX'

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

def ortho_basis(arg, deg):
    nelems = deg + 1
    assert nelems > 0
    x = arg
    domain = 'EX'
    basis = [sy.Poly(1, x, domain=domain)]
    for i in range(1, nelems):
        q = sy.Poly(0, x, domain=domain)
        p = sy.Poly(x**i, x, domain=domain)
        for j in range(i):
            e = prod(p, basis[j]) * basis[j]
            e,_ = e.div(norm_sq(basis[j]))
            q += e
        b = p - q
        basis += [b]
    return basis

def test_ortho_basis():
    x = sy.symbols('x', real=True)
    basis = ortho_basis(x, 10)

    for i,b in enumerate(basis):
        print(f'b{i} = {b}')

    for i in range(1, len(basis)):
        for j in range(len(basis)):
            v = prod(basis[i], basis[j])
            if i == j:
                assert v > 0
            else:
                assert v == 0

def test_orthonormal_basis():
    x = sy.symbols('x', real=True)
    basis = find_orthonormal_basis(x, 10)

    for i,b in enumerate(basis):
        print(f'b{i} = {b}')

    for i in range(1, len(basis)):
        for j in range(len(basis)):
            d = prod(basis[i], basis[j])
            assert d == ((i == j) * 1)

def plot_basis():
    x = sy.symbols('x', real=True)
    basis = find_orthonormal_basis(x, 10)
    basis = [sy.lambdify(x, f.expr) for f in basis]

    x1, x2 = get_limits()
    x = np.linspace(x1, x2, 1000)

    y = np.zeros((len(x), len(basis)))
    for i,f in enumerate(basis):
        y[:,i] = f(x)
    
    plt.plot(x, y)
    plt.grid()
    plt.show()

def decompose(f, basis):
    coefs = []
    r = f
    for i in range(len(basis) - 1, -1, -1):
        c, r = r.div(basis[i])
        coefs += [c]
    return coefs[::-1]

def collocation_points():
    x = sy.symbols('x', real=True)
    n = 20
    basis = find_orthonormal_basis(x, n)

    coefs = []

    for k in range(1, n-1):
        c = decompose(basis[k] * x, basis)
        coefs += [c[k-1].expr]

    for i in range(1, len(coefs)):
        d = coefs[i] / coefs[i-1]
        print(i, d**2)
    
    # plt.plot(s, 'o')
    # plt.plot([0, len(s)-1], [], '-')
    # plt.show()

    # k1 = c[k-1].expr
    # k2 = c[k+1].expr
    # sy.pprint(1/k1)
    # sy.pprint(k2/k1)

def test():
    s = np.array([4, 27, 80, 175, 324, 539, 832, 1215, 1700, 2299, 3024, 3887, 4900, 6075, 7424, 8959, 10692])
    s = np.cbrt(s)
    k = (s[-1] - s[0]) / (len(s) - 1)

    t = np.linspace(-5, len(s)+5)

    plt.plot(s)
    plt.plot(t, k*t + s[0])
    plt.grid()
    plt.show()

def test2():
    x = sy.symbols('x')
    _1 = sy.Poly(1, x)
    basis = find_orthonormal_basis(x, 7)
    f = sy.Poly(x**5, x)
    coefs = decompose(f, basis)
    for c in coefs:
        sy.pprint(c.expr)
    
    sy.pprint(basis[-1].expr)

def test3():
    x = sy.symbols('x', real=True)
    basis = find_orthonormal_basis(x, 20)

    n = 3
    m = 7
    bn = basis[n]
    bm = basis[m]

    f = bn * bm
    coefs = decompose(f, basis)
    for c in coefs:
        sy.pprint(c.expr)

    # for k in range(0, n):
    #     arg = sy.cos(sy.pi * (2 * k + 1) / (2 * n + 1))
    #     v = bn(arg)
    #     sy.pprint(v)

def decompose_deriv(k):
    '''
        decompose d bk(x) / dx
    '''
    d = np.zeros(k)
    n0 = (k + 1) % 2
    for n in range(n0, k, 2):
        v = (2*k + 1) * (2*n + 1)
        d[n] = np.sqrt(v)
    return d

def test4():
    x = sy.symbols('x', real=True)
    k = 12
    basis = find_orthonormal_basis(x, 15)
    bk = basis[k]
    Dbk = bk.diff(x)
    # coefs = decompose(Dbk, basis)
    # coefs = np.array([e.expr for e in coefs], float)
    # d = decompose_deriv(k)
    # print(d)

if __name__ == '__main__':
    # test_ortho_basis()
    test_orthonormal_basis()
    # collocation_points()
    # plot_basis()
    # test2()
    # test3()
    # find_orthonormal_basis()
    # test4()
