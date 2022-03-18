import numpy as np
from scipy.integrate import quad


def get_limits():
    return -1,1

def product_poly(p1, p2):
    p = np.polymul(p1, p2)
    Ip = np.polyint(p)
    x1,x2 = get_limits()
    Il = np.polyval(Ip, x1)
    Ir = np.polyval(Ip, x2)
    return Ir - Il

def product_fun(f1, f2):
    f = lambda x: f1(x) * f2(x)
    I,*_ = quad(f, *get_limits())
    return I

def product(f1, f2):
    if isinstance(f1, np.ndarray) and isinstance(f2, np.ndarray):
        return product_poly(f1, f2)
    _f1 = (lambda x: np.polyval(f1, x))   if isinstance(f1, np.ndarray) else f1
    _f2 = (lambda x: np.polyval(f2, x))   if isinstance(f2, np.ndarray) else f2
    return product_fun(_f1, _f2)

def norm_sq(p):
    return product(p, p)

def norm(p):
    return np.sqrt(norm_sq(p))


def __orthonormal_basis(deg):
    assert deg > 0

    b0 = np.array([1.])
    b0 /= norm(b0)
    basis = [b0]

    for i in range(1, deg + 1):
        p = np.zeros(i + 1)
        p[0] = 1
        q = np.array([0])
        for j in range(i):
            q = np.polyadd(q, product(p, basis[j]) * basis[j])
        b = np.polysub(p, q)
        b = b / norm(b)
        basis += [b]
    
    return basis

def orthonormal_basis(deg):
    assert deg > 0

    b0 = np.array([np.sqrt(2)/2])
    b1 = np.array([np.sqrt(6)/2, 0])
    basis = [b0, b1]

    for n in range(2, deg + 1):
        b1 = basis[-1]
        b2 = basis[-2]
        u0 = n / np.sqrt(4*n**2 - 1)
        u1 = (n - 1) / np.sqrt(4*(n - 1)**2 - 1)
        b = np.polysub(
            np.polymul([1 / u0, 0], b1), 
            b2 * u1 / u0
        )
        basis += [b]

    return basis

def orthogonal_basis(deg):
    assert deg > 0

    b0 = np.array([1.])
    basis = [b0]

    for i in range(1, deg + 1):
        p = np.zeros(i + 1)
        p[0] = 1
        q = np.array([0])
        for j in range(i):
            q = np.polyadd(q, product(p, basis[j]) * basis[j] / norm_sq(basis[j]))
        b = np.polysub(p, q)
        basis += [b.copy()]
    
    return basis

def decompose(p, basis):
    return [product(p, b) / norm(b) for b in basis]

def test_orthogonal_basis():
    deg = 15
    basis = orthogonal_basis(deg)
    nelems = len(basis)
    assert nelems == deg + 1
    for i in range(nelems):
        for j in range(nelems):
            s = product(basis[i], basis[j])
            assert np.abs(s) < 1e-12 or i == j

def test_orthonormal_basis():
    deg = 15
    basis = orthonormal_basis(deg)
    nelems = len(basis)
    assert nelems == deg + 1
    for i in range(nelems):
        for j in range(nelems):
            s = product(basis[i], basis[j])
            assert np.allclose(s, int(i == j), atol=1e-7)

def test_decompose():
    deg = 5
    basis = orthonormal_basis(deg)
    poly = np.random.normal(size=deg+1)
    coefs = decompose(poly, basis)

    q = np.array([0])
    for c,b in zip(coefs,basis):
        q = np.polyadd(q, c * b)

    assert np.allclose(poly, q)

def deriv_ceffs(deg):
    '''
        decompose d bk(x) / dx
    '''
    d = np.zeros(deg)
    n0 = (deg + 1) % 2
    for n in range(n0, deg, 2):
        v = (2*deg + 1) * (2*n + 1)
        d[n] = np.sqrt(v)
    return d

def test_deriv_ceffs():
    deg = 14
    d = deriv_ceffs(deg)
    basis = orthonormal_basis(deg)
    bk = basis[-1]
    Dbk = np.polyder(bk)
    coefs = decompose(Dbk, basis)
    assert np.allclose(d, coefs[:-1])

def basis_mat_form(basis):
    R'''
        Form a matrix from polynomial coefficients
    '''
    n = len(basis)
    B = np.zeros((n, n))
    for i in range(n):
        B[i, n-i-1:] = basis[i]
    return B

def deriv_mat(deg):
    n = deg + 1
    D = np.zeros((n, n))
    for k in range(1, deg + 1):
        dk = deriv_ceffs(k)
        D[k,:k] = dk
    return D

def test_orthonormal_basis_mat():
    deg = 6
    basis = orthonormal_basis(deg)
    B = basis_mat_form(basis)
    print(B)

def eval(B, c, x):
    return np.polyval(c @ B, x)

def test_deriv_mat():
    deg = 7
    basis = orthonormal_basis(deg)
    B = basis_mat_form(basis)
    D = deriv_mat(deg)
    DB = D @ B

    c = np.random.normal(size=(deg+1))
    x = np.random.normal()
    val1 = eval(DB, c, x)

    # on the other hand
    basis = orthonormal_basis(deg)
    q = np.array([], float)
    for b,c in zip(basis, c):
        q = np.polyadd(q, np.polyder(b) * c)
    val2 = np.polyval(q, x)

    assert np.allclose(val1, val2)

def fit(x, y, deg):
    basis = orthonormal_basis(deg)
    n = deg + 1
    nx = len(x)
    A = np.zeros((nx, n))
    for i,b in enumerate(basis):
        A[:,i] = np.polyval(b, x)
    c,*_ = np.linalg.lstsq(A, y, rcond=-1)
    return c

def test_fit():
    import matplotlib.pyplot as plt
    
    # v1
    deg = 7
    x = np.linspace(-1, 1, 100)
    y = np.sin(x)
    c = fit(x, y, deg)
    basis = orthonormal_basis(deg)
    B = basis_mat_form(basis)
    y1 = eval(B, c, x)
    plt.plot(x, y1 - y)
    plt.show()

def test_decompose_2():
    import matplotlib.pyplot as plt

    f = lambda x: np.sin(x)
    deg = 11
    basis = orthonormal_basis(deg)
    coefs = decompose(f, basis)
    B = basis_mat_form(basis)
    D = deriv_mat(deg)
    DB = D @ B
    xx = np.linspace(-1,1,1000)
    yy1 = eval(DB, coefs, xx)
    yy2 = np.cos(xx)
    plt.plot(xx, yy1 - yy2)
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=200)
    # test_deriv_mat()
    # test_orthonormal_basis_mat()
    # test_orthogonal_basis()
    # test_orthonormal_basis()
    # test_decompose()
    # test_deriv_ceffs()
    # test_fit()
    test_decompose_2()