# solving subproblem \ref{eq: sub_sim}
# min g^\top x + \frac{1}{2} x^\top B x + \theta(x)

import numpy as np
import time
from auxiliary_functions import B_alpha_z_direct, B_alpha_inv_z_direct, B_alpha_inv_D_inv_z_direct, initialize_lamb_direct
from problem_setup import prox_h
from stopping import check_stopping, check_stopping_nonsmooth


def ssn_direct(v, bfgs, x_current, eta=1e-2, lam=1e-3, mu=1e-3, tol=1e-2, max_iter=100, stopping='our', ratio=1e-2):
    '''
    v: gradient estimation at x_k
    bfgs: BFGS object
    x_current: current point x_k
    eta: learning rate
    tol: t in the function check_stopping_nonsmooth(), proportional to eta
    '''
    # choose alpha
    alpha_inv = 1 / bfgs.sigma
    for i in range(bfgs.k_size):
        alpha_inv += bfgs.s[i].dot(bfgs.s[i]) / bfgs.y[i].dot(bfgs.s[i])
    alpha = 0.5 / alpha_inv

    # compute big_SY, big_SYT, K_0, K_k
    S = np.array(bfgs.s).T
    Y = np.array(bfgs.y).T
    bfgs.big_SY = np.concatenate((bfgs.sigma * S, Y), axis=1)
    bfgs.big_SYT = bfgs.big_SY.T
    S_S = S.T.dot(S)  # pretty fast
    S_Y = S.T.dot(Y)
    Y_Y = Y.T.dot(Y)
    L_SY = np.tril(S_Y, k=-1)
    D = np.diag(np.diag(S_Y))
    bfgs.K_0 = np.block([[bfgs.sigma * S_S, L_SY], [L_SY.T, -D]])
    block1 = bfgs.sigma * alpha * S_S / (bfgs.sigma - alpha)
    block2 = bfgs.sigma * S_Y / (bfgs.sigma - alpha) - L_SY
    block3 = Y_Y / (bfgs.sigma - alpha) + D
    bfgs.K_k = np.block([[block1, block2], [block2.T, block3]])

    # initialization
    g = eta * v - B_alpha_z_direct(x_current, alpha, bfgs) - alpha * x_current
    reg = eta * lam
    lamb0 = 0.01 * np.ones_like(g)
    start_time = time.time()
    lamb = initialize_lamb_direct(lamb0, g, alpha, reg, bfgs, ratio)[0]
    grad_init = B_alpha_inv_z_direct(lamb - g, alpha, bfgs) - prox_h(-lamb / alpha, reg / alpha)
    end_time = time.time()

    acc_time = end_time - start_time
    ave_search_length = 0

    def grad(x):
        return v + B_alpha_z_direct(x - x_current, 0, bfgs)/eta
    # inner loop
    for iter in range(max_iter):
        x = B_alpha_inv_z_direct(lamb - g, alpha, bfgs)
        z = prox_h(-lamb / alpha, reg / alpha)
        grad_Lamb = x - z
        d = B_alpha_inv_D_inv_z_direct(grad_Lamb, lamb, alpha, reg, bfgs)
        if stopping == 'our':
            if check_stopping(grad, x, lam, tol * mu): # stopping criterion (revise)
                break
        else:
            if np.linalg.norm(grad_Lamb)/np.linalg.norm(grad_init) < tol:
                break

        # need to compute \Lambda(lamb - rho * d) - \Lambda(lamb), refer to formula, store some terms
        B_alpha_inv_d = B_alpha_inv_z_direct(d, alpha, bfgs)
        term1 = d.dot(B_alpha_inv_d) / 2 + d.dot(d) / (2 * alpha)  # need to scaled by rho^2
        term2 = - (lamb - g).dot(B_alpha_inv_d) - lamb.dot(d) / alpha  # need to scaled by rho

        temp11 = - lamb / alpha
        temp12 = np.sign(temp11) * np.maximum(np.abs(temp11) - reg / alpha, 0)
        Moreau1 = (temp12 - temp11).dot(temp12 - temp11) / 2 + reg / alpha * np.linalg.norm(temp12, ord=1)
        temp21 = - lamb / alpha
        temp22 = np.sign(temp21) * np.maximum(np.abs(temp21) - reg / alpha, 0)
        Moreau2 = (temp22 - temp21).dot(temp22 - temp21) / 2 + reg / alpha * np.linalg.norm(temp22, ord=1)

        # line search
        rho_test = 1
        for it in range(20):
            lamb_test = lamb - rho_test * d

            # compute \Lambda(lamb - rho * d) - \Lambda(lamb)
            Lamb_gap = rho_test * rho_test * term1 + rho_test * term2 - alpha * (Moreau1 - Moreau2)
            RHS = - 1e-4 * rho_test * d.dot(grad_Lamb)

            if Lamb_gap < RHS:
                break
            else:
                rho_test *= 0.9
        lamb = lamb_test.copy()
        ave_search_length += it
    ave_search_length /= (iter+1)
        # print(rho_test, iter)
    # print('num of inner iterations for direct method:', j, '\n')
    # print(Lambda(lamb, g, alpha, reg, bfgs))
    # print('num of iter:', iter)
    # print('avg linesearch:', counter)
    # print('time for calculating objective Lambda:', acc_time)

    return x, [acc_time, ave_search_length, iter]


# testing the ssn_direct
if __name__ == "__main__":
    # f = 0
    from BFGS_class import BFGS

    d = 10000
    np.random.seed(0)
    x = np.random.randn(d)
    v = np.random.randn(d)
    x_current = np.random.randn(d)
    bfgs = BFGS()
    for i in range(10):
        x_old = x.copy()
        x -= 0.1 * x
        bfgs.s.append(x - x_old)
        bfgs.y.append(x - x_old)
        bfgs.k_size += 1
    bfgs.sigma = bfgs.y[-1].dot(bfgs.y[-1]) / bfgs.s[-1].dot(bfgs.y[-1])
    # compute auxiliary matrices
    S = np.array(bfgs.s).T
    Y = np.array(bfgs.y).T
    bfgs.big_SY = np.concatenate((bfgs.sigma * S, Y), axis=1)
    bfgs.big_SYT = bfgs.big_SY.copy().T
    S_S = S.T.dot(S)  # pretty fast
    S_Y = S.T.dot(Y)
    Y_Y = Y.T.dot(Y)
    L_SY = np.tril(S_Y, k=-1)
    D = np.diag(np.diag(S_Y))
    bfgs.K_0 = np.block([[bfgs.sigma * S_S, L_SY], [L_SY.T, -D]])

    tic = time.time()
    output = ssn_direct(v, bfgs, x_current, 0.1, tol=1e-10)
    toc = time.time()
    print('time:', toc - tic)
    print(output[0])
