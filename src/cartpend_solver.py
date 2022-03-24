from cartpend import Dynamics
from bvp_test import solve_bvp


def solve():
    p = Parameters(m_pend = 0.1, m_cart=0.5, l = 0.5, g = 9.8)
    d = Dynamics(p)
    solve_bvp