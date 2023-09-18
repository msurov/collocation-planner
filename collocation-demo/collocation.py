from casadi import pi, SX, vertcat, Function
from numpy.polynomial.legendre import legder, legroots, legval
import numpy as np


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
    L = np.zeros(N)
    L[-1] = 1
    DL = legder(L)
    roots = legroots(DL)
    assert len(roots) == N-2
    assert np.allclose(legval(roots, DL), 0)
    roots = np.sort(roots)
    collocation_points = np.concatenate(([-1], roots, [1]))
    return collocation_points

def get_lgl_weights(collocation_points):
    N = len(collocation_points)
    L = np.zeros(N)
    L[-1] = 1
    weights = np.zeros(N)
    for j in range(N):
        tmp = N * (N-1) * legval(collocation_points[j], L)**2
        weights[j] = 2 / tmp
    return weights

def get_cgl_collocation_points(N):
    i = np.arange(N)
    cp = -np.cos(np.pi * i / (N - 1))
    return cp

def get_cgl_weights(collocation_points):
    N = len(collocation_points)
    weights = pi/(N - 1) * np.ones(N)
    weights[0] = pi/(2*N - 2)
    weights[-1] = pi/(2*N - 2)
    return weights

def test_lgl():
    N = 23
    collocation_points = get_lgl_collocation_points(N)
    weights = get_lgl_weights(collocation_points)
    I1 = np.exp(2 * collocation_points) @ weights
    I2 = 0.5 * np.exp(2 * 1) - 0.5 * np.exp(2 * -1)
    assert np.allclose(I1, I2)

def test_cgl():
    N = 25
    collocation_points = get_cgl_collocation_points(N)
    weights = get_cgl_weights(collocation_points)
    I1 = 0
    for cp,w in zip(collocation_points, weights):
        I1 += np.sqrt(1 - cp**2) * np.exp(2 * cp) * w
    I2 = 0.5 * np.exp(2 * 1) - 0.5 * np.exp(2 * -1)
    print(I1)
    print(I2)

if __name__ == '__main__':
    test_lgl()
