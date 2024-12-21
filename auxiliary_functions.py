import numpy as np
import time


def prox_h(w, lam=0.01):  # proximal operator of lam * L1_norm
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)


def calculate_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print("function", func.__name__, "execution time is:", execution_time, "s")
        return result
    return wrapper



# auxiliary functions for direct method
def B_alpha_z_direct(z, alpha, bfgs):
    res = (bfgs.sigma - alpha) * z
    temp1 = bfgs.big_SYT.dot(z)
    temp2 = np.linalg.solve(bfgs.K_0, temp1)
    res -= bfgs.big_SY.dot(temp2)
    return res


def B_alpha_inv_z_direct(z, alpha, bfgs):
    res = z / (bfgs.sigma - alpha)
    temp1 = bfgs.big_SYT.dot(res)
    temp2 = np.linalg.solve(bfgs.K_k, temp1)
    res -= bfgs.big_SY.dot(temp2) / (bfgs.sigma - alpha)
    return res


def C_alpha_z(z, lamb, alpha, reg, bfgs):
    comparison = np.abs(lamb) > reg
    index = np.where(comparison)[0]
    res = (bfgs.sigma - alpha) * z
    if z.ndim == 1:
        res[index] *= alpha / bfgs.sigma
    elif z.ndim == 2:
        for j in range(z.shape[1]):
            res[:, j][index] *= alpha / bfgs.sigma
    else:
        raise ValueError('too large dimension of z')
    return res


def B_alpha_inv_D_inv_z_direct(z, lamb, alpha, reg, bfgs):
    res = C_alpha_z(z, lamb, alpha, reg, bfgs)
    temp1 = bfgs.big_SYT.dot(res) / (bfgs.sigma - alpha)
    bar_K_k = bfgs.big_SYT.dot(C_alpha_z(bfgs.big_SY, lamb, alpha, reg, bfgs)) / (bfgs.sigma - alpha) ** 2
    K_bar_K = bar_K_k - bfgs.K_k
    temp2 = np.linalg.solve(K_bar_K, temp1)
    temp3 = bfgs.big_SY.dot(temp2) / (bfgs.sigma - alpha)
    res -= C_alpha_z(temp3, lamb, alpha, reg, bfgs)
    return res


# calculate Lambda(lamb)
def Lambda_direct(lamb, g, alpha, reg, bfgs):
    temp1 = lamb - g
    res1 = temp1.dot(B_alpha_inv_z_direct(temp1, alpha, bfgs)) / 2 + lamb.dot(lamb) / (2 * alpha)
    # print('res1 = ', res1)
    temp2 = - lamb / alpha
    temp3 = np.sign(temp2) * np.maximum(np.abs(temp2) - reg / alpha, 0)
    res2 = (temp3 - temp2).dot(temp3 - temp2) / 2 + reg / alpha * np.linalg.norm(temp3, ord=1)
    res = res1 - alpha * res2
    return res

def initialize_lamb_direct(lamb, g, alpha, reg, bfgs, ratio=1e-2):
    # grad_lamb = B_alpha_inv_z_direct(lamb - g, alpha, bfgs) - prox_h(-lamb / alpha, reg / alpha)
    # comparison = ~(np.abs(lamb) > reg)
    # temp1 = comparison * grad_lamb / alpha
    # temp2 = B_alpha_inv_z_direct(grad_lamb, alpha, bfgs) + temp1
    # temp3 = temp2.dot(grad_lamb)
    # t = grad_lamb.dot(grad_lamb) / temp3
    # print(t)
    # res = lamb - t * grad_lamb
    max_iter = 1000
    alpha_inv = 1 / bfgs.sigma
    for i in range(bfgs.k_size):
        alpha_inv += bfgs.s[i].dot(bfgs.s[i]) / bfgs.y[i].dot(bfgs.s[i])
    alpha_bar = 1 / alpha_inv
    L = 1 / (alpha_bar - alpha) + 1 / alpha

    obj_old = Lambda_direct(lamb, g, alpha, reg, bfgs)
    grad_lamb = B_alpha_inv_z_direct(lamb - g, alpha, bfgs) - prox_h(-lamb / alpha, reg / alpha)
    res = lamb - 1 / L * grad_lamb
    obj_new = Lambda_direct(res, g, alpha, reg, bfgs)
    gap_0 = np.abs(obj_new - obj_old)

    for iter in range(max_iter):
        obj_old = obj_new.copy()
        grad_lamb = B_alpha_inv_z_direct(res - g, alpha, bfgs) - prox_h(-res / alpha, reg / alpha)
        res -= 1 / L * grad_lamb
        obj_new = Lambda_direct(res, g, alpha, reg, bfgs)
        # print(res)
        gap = np.abs(obj_new - obj_old)
        if gap/gap_0 < ratio:
            break
    return res, iter