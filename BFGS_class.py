from collections import deque
import numpy as np


class BFGS:  # to manage BFGS related attributes
    def __init__(self, k_max=10):
        self.k_max = k_max
        self.s = deque(maxlen=k_max)
        self.y = deque(maxlen=k_max)
        self.k_size = 0  # current size
        self.sigma = None
        self.big_SY = None
        self.big_SYT = None
        self.K_0 = None
        self.K_k = None
        # self.K_alpha = None
        # self.bar_K_alpha = None
        # self.c_alpha = None
        self.a = None
        self.b = None
        self.p = None
        self.p_half = None
        self.nu = None
        self.nu_half = None

def LBFGS(z, bfgs):  # bfgs is a realization of class BFGS
    # realization of inverse Hessian-vector product with H generated from Algo 2 (Hessian Update)
    q = z
    alpha = []
    m = bfgs.k_size
    for i in range(1, m + 1):
        rho = 1 / np.dot(bfgs.y[-i], bfgs.s[-i])
        alpha.append(rho * np.matmul(bfgs.s[-i].T, q))
        q = q - alpha[-1] * bfgs.y[-i]
    r = 1 / bfgs.sigma * q
    alpha.reverse()
    for j in range(m, 0, -1):
        rho = 1 / np.dot(bfgs.y[-j], bfgs.s[-j])
        beta = rho * np.dot(bfgs.y[-j].T, r)
        r = r + (alpha[-j] - beta) * bfgs.s[-j]
    return r
