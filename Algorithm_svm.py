import numpy as np
from tqdm import tqdm
from time import time
from scipy.sparse.linalg import norm

from BFGS_class import BFGS
from problem_setup import f_svm_grad, F_svm, prox_h, f_svm, f_svm_hess_z
from fista import fista_backtracking, fista
from CG_method import conjugate_gradient
from ssn_direct import ssn_direct


def ISPQN_svm(X, y, w_init=None, lr=1e-3, lam=1e-3, mu=1e-3, inner_iter=100, outer_iter=100, opts={}):
    '''
    Inexact Stochastic Proximal Quasi-Newton method
    grad: gradient function
    b: batch size
    H0: H_0 = H0 I
    w_init: initial point
    lr: learning rate
    lam: regularization parameter
    mu: strong convexity parameter
    iter_num: number of iterations
    opts: options_dict
    '''
    opts.setdefault('grad_scheme', 'svrg')
    opts.setdefault('stopping', 'our')
    opts.setdefault('inner_tol', 1e-2)
    opts.setdefault('inner_solver', 'fista')
    opts.setdefault('inner_max', 100)
    opts.setdefault('Hessian', 'sdlbfgs') # stochastic L-BFGS in bryd's paper
    opts.setdefault('M', 10) # memory size
    opts.setdefault('r', 10) # Hessian update frequency
    opts.setdefault('b', 100) # batch size
    opts.setdefault('window_size', 10)

    d = X.shape[1]
    if w_init is None:
        w_init = 0.1 * np.ones(d)

    if lam == 0:
        return ISPQN_svm_smooth(X, y, w_init, lr, mu, inner_iter, outer_iter, opts)
    else:
        return ISPQN_svm_nonsmooth(X, y, w_init, lr, lam, mu, inner_iter, outer_iter, opts)

def ISPQN_svm_smooth(X, y, w_init=None, learning_rate=1e-2, mu=1e-3, inner_iter=10, outer_iter=10, opts=None):
    '''subproblem is solved by CG'''
    L = opts['r']
    M = opts['M']
    b = opts['b']
    n, d = X.shape
    bfgs = BFGS(M)
    w_hat = w_init.copy() # reference point

    # initialize settings for slbfgs
    w_bar_old = np.zeros(d)
    w_bar_new = np.zeros(d)

    # initialize settings for algorithm
    w = w_init.copy()
    n_epochs = inner_iter * outer_iter
    loss = np.zeros(n_epochs)
    w_sequence = np.zeros((n_epochs, d))
    error_sequence = np.zeros(n_epochs)

    center_grad = f_svm_grad(w_hat, X, y, mu)
    lr = learning_rate

    max_inner_iter = 0
    ave_inner_iter = 0
    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            if opts['grad_scheme'] == 'sg' and (k*inner_iter+j) % opts['window_size'] == 0:
                lr /= 2
            p = np.random.permutation(n)

            # store current loss
            loss[k * inner_iter + j] = f_svm(w, X, y, mu)
            w_sequence[k * inner_iter + j] = w.copy()
            error_sequence[k * inner_iter + j] = np.linalg.norm(f_svm_grad(w, X, y, mu))

            # update correction pairs bfgs.s and bfgs.y according to SdLBFGS
            if (k * inner_iter + j) % L == 0 and k * inner_iter + j > 0:
                w_bar_new /= L
                s_new = w_bar_new - w_bar_old
                y_new = f_svm_grad(w_bar_new, X[p[:b]], y[p[:b]], mu) - f_svm_grad(w_bar_old, X[p[:b]], y[p[:b]], mu)
                # update bfgs.sigma
                s_y = np.dot(s_new, y_new)
                bfgs.sigma = max(np.linalg.norm(y_new) ** 2 / s_y, 1e-1) # ensure sigma > 0
                ss_sigma = np.dot(s_new, s_new) * bfgs.sigma
                theta = 1
                if s_y < 0.25 * ss_sigma:
                    theta = 0.75 * ss_sigma / (ss_sigma-s_y)
                y_new = theta * y_new + (1-theta) * bfgs.sigma * s_new

                w_bar_old = w_bar_new.copy()
                w_bar_new = np.zeros(d)

                if bfgs.k_size < M:
                    bfgs.s.append(s_new)
                    bfgs.y.append(y_new)
                    bfgs.k_size += 1
                else:
                    bfgs.s.popleft()
                    bfgs.y.popleft()
                    bfgs.s.append(s_new)
                    bfgs.y.append(y_new)

            if opts['grad_scheme'] == 'svrg':
                # compute variance reduced stochastic gradient
                grad = f_svm_grad(w, X[p[:b]], y[p[:b]], mu) - f_svm_grad(w_hat, X[p[:b]], y[p[:b]], mu) + center_grad
            else:
                grad = f_svm_grad(w, X[p[:b]], y[p[:b]], mu)

            w_bar_new += w

            # update w
            if k * inner_iter + j < L:
                w = w - lr * grad

            else:    
                # compute big_SY, big_SYT, K_0, K_k
                S = np.array(bfgs.s).T
                Y = np.array(bfgs.y).T
                bfgs.big_SY = np.concatenate((bfgs.sigma * S, Y), axis=1)
                bfgs.big_SYT = bfgs.big_SY.T
                S_S = S.T.dot(S)  # pretty fast
                S_Y = S.T.dot(Y)
                L_SY = np.tril(S_Y, k=-1)
                D = np.diag(np.diag(S_Y))
                bfgs.K_0 = np.block([[bfgs.sigma * S_S, L_SY], [L_SY.T, -D]])
                w, iter = conjugate_gradient(grad, lr, bfgs, w, opts['inner_max'], opts['inner_tol'], stopping=opts['stopping']) # solve subproblem
                max_inner_iter = max(max_inner_iter, iter)
                ave_inner_iter += iter
            time_list.append(time()-tic)
        w_hat = w.copy()
        center_grad = f_svm_grad(w_hat, X, y, mu)
    ave_inner_iter /= (outer_iter * inner_iter)
    return loss, w_sequence, error_sequence, [max_inner_iter, ave_inner_iter], time_list[:-1]
    
def ISPQN_svm_nonsmooth(X, y, w_init=None, learning_rate=1e-2, lam=1e-3, mu=1e-3, inner_iter=10, outer_iter=10, opts=None):
    M = opts['M']
    # L = opts['r']
    b = opts['b']
    n, d = X.shape
    bfgs = BFGS(M)
    w_hat = w_init.copy() # reference point

    # initialize settings for algorithm
    # w = w_init.copy()
    w_old = w_init.copy() # x_{k-1}
    w_curr = w_init.copy() # current x_k
    n_epochs = inner_iter * outer_iter
    loss = np.zeros(n_epochs)
    w_sequence = np.zeros((n_epochs, d))
    error_sequence = np.zeros(n_epochs)

    center_grad = f_svm_grad(w_hat, X, y, mu)
    lr = learning_rate

    acc_time = 0
    max_inner_iter = 0
    max_ave_search_len = 0
    ave_inner_iter = 0
    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            if opts['grad_scheme'] == 'sg' and (k*inner_iter+j) % opts['window_size'] == 0:
                lr /= 2
            p = np.random.permutation(n)

            # store current loss
            loss[k * inner_iter + j] = F_svm(w_curr, X, y, lam, mu)
            w_sequence[k * inner_iter + j] = w_curr.copy()
            error_sequence[k * inner_iter + j] = np.linalg.norm(w_curr - prox_h(w_curr-f_svm_grad(w_curr, X, y, mu), lam))

            # update correction pairs bfgs.s and bfgs.y according to SdLBFGS
            if k * inner_iter + j > 0:
                s_new = w_curr - w_old
                y_new = f_svm_grad(w_curr, X[p[:b]], y[p[:b]], mu) - f_svm_grad(w_old, X[p[:b]], y[p[:b]], mu)
                # update bfgs.sigma
                s_y = np.dot(s_new, y_new)
                bfgs.sigma = max(np.linalg.norm(y_new) ** 2 / s_y, 1e-1) # ensure sigma > 0
                ss_sigma = np.dot(s_new, s_new) * bfgs.sigma
                theta = 1
                if s_y < 0.25 * ss_sigma:
                    theta = 0.75 * ss_sigma / (ss_sigma-s_y)
                y_new = theta * y_new + (1-theta) * bfgs.sigma * s_new

                if bfgs.k_size < M:
                    bfgs.s.append(s_new)
                    bfgs.y.append(y_new)
                    bfgs.k_size += 1
                else:
                    bfgs.s.popleft()
                    bfgs.y.popleft()
                    bfgs.s.append(s_new)
                    bfgs.y.append(y_new)
                w_old = w_curr.copy()

            if opts['grad_scheme'] == 'svrg':
                # compute variance reduced stochastic gradient
                grad = f_svm_grad(w_curr, X[p[:b]], y[p[:b]], mu) - f_svm_grad(w_hat, X[p[:b]], y[p[:b]], mu) + center_grad
            else:
                grad = f_svm_grad(w_curr, X[p[:b]], y[p[:b]], mu)

            # update w
            if k * inner_iter + j == 0:
                w_curr = prox_h(w_curr - lr * grad, lr * lam)

            else:
                if opts['inner_solver'] == 'fista':
                    # update bfgs information, compute big_SY, big_SYT, K_0, K_k
                    S = np.array(bfgs.s).T
                    Y = np.array(bfgs.y).T
                    bfgs.big_SY = np.concatenate((bfgs.sigma * S, Y), axis=1)
                    bfgs.big_SYT = bfgs.big_SY.T
                    S_S = S.T.dot(S)  # pretty fast
                    S_Y = S.T.dot(Y)
                    L_SY = np.tril(S_Y, k=-1)
                    D = np.diag(np.diag(S_Y))
                    bfgs.K_0 = np.block([[bfgs.sigma * S_S, L_SY], [L_SY.T, -D]])
                    #update w_k
                    w_curr, iter = fista_backtracking(grad, bfgs, w_curr, lr, lam, tol=opts['inner_tol'], L0=1, max_iter=opts['inner_max'], stopping=opts['stopping'])
                    max_inner_iter = max(max_inner_iter, iter)
                    ave_inner_iter += iter
                elif opts['inner_solver'] == 'ssn':
                    w_curr, info = ssn_direct(grad, bfgs, w_curr, lr, lam, mu, max_iter=opts['inner_max'], tol=opts['inner_tol'], stopping=opts['stopping'], ratio=opts['ssn_ratio']) # solve subproblem
                    acc_time += info[0]
                    max_inner_iter = max(max_inner_iter, info[1])
                    max_ave_search_len = max(max_ave_search_len, info[2])
                    ssn_info = [acc_time, max_inner_iter, max_ave_search_len]
                else:
                    raise NotImplementedError('optimizer {} is not implemented'.format(opts['inner_solver']))
            time_list.append(time()-tic)
        w_hat = w_curr.copy()
        center_grad = f_svm_grad(w_hat, X, y, mu)
    ave_inner_iter /= (outer_iter * inner_iter)
    if opts['inner_solver'] == 'ssn':
        return loss, w_sequence, error_sequence, ssn_info, time_list[:-1]
    else:
        return loss, w_sequence, error_sequence, [max_inner_iter, ave_inner_iter], time_list[:-1]
    
def subnewton_svm_nonsmooth(X, y, w_init=None, lr=1e-3, lam=1e-3, mu=1e-3, inner_iter=10, outer_iter=10, opts=None):
    '''approximate proximal subsampled Newton method for regularized logistic regression
        problem setup: 
            X: feature matrix
            y: labels 
            mu: l_2 regularization parameter
            lam: l_1 regularization parameter
        algorithm setup:
            w_init: initial point
            lr: learning rate
            iter_num: number of iterations
            opts: options for the algorithm
    '''
    # initilization
    n, d = X.shape
    b = opts['b']
    bH = opts['b_H']

    l_r = lr
    
    if w_init is None:
        w_init = 0.01 * np.ones(d)
    w_old = w_init.copy()
    w_hat = w_init.copy() # reference point
    center_grad = f_svm_grad(w_hat, X, y, mu)

    # store sequences
    iter_num = outer_iter * inner_iter
    w_sequence = np.zeros((iter_num, d)) # store x_k
    loss_sequence = np.zeros(iter_num) # store F(x_k)
    inner_iter_nums = np.zeros(iter_num) # store number of inner iterations to update x_k
    error_sequence = np.zeros(iter_num)

    flag_dense = isinstance(X, np.ndarray)

    inner_opts = {'stopping': opts['stopping'], 'max_iter': opts['inner_max']}
    time_list = [0]


    tic=time()

    # main loop
    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            w_sequence[k * inner_iter + j] = w_old
            loss_sequence[k * inner_iter + j] = F_svm(w_old, X, y, lam, mu)
            error_sequence[k * inner_iter + j] = np.linalg.norm(w_old - prox_h(w_old-f_svm_grad(w_old, X, y, mu), lam))

            # select independent samples O_k and S_k
            p1 = np.random.permutation(n)
            p2 = np.random.permutation(n)
            O = p1[:b]
            S = p2[:bH]

            # Compute gradient approximation g_k
            if opts['grad_scheme'] == 'svrg':
                g_k = f_svm_grad(w_old, X[O], y[O], mu) - f_svm_grad(w_hat, X[O], y[O], mu) + center_grad
            else:
                g_k = f_svm_grad(w_old, X[O], y[O], mu)

            # use np.linalg.norm() for narray and norm() for sparse matrix (very important!!!)
            if flag_dense:
                L = 2 * np.max(np.linalg.norm(X[S], axis=1, ord=2)) + mu
            else: # csr_sparse matrix
                L = 2 * np.max(norm(X[S], axis=1, ord=2)) + mu
            def grad(w): # the gradient of inner FISTA, linear in w
                return g_k + f_svm_hess_z(w_old, X[S], y[S], w-w_old, mu)
            
            # define options for inner FISTA
            if inner_opts['stopping'] == 'our':
                inner_opts['threshold'] = opts['inner_tol'] * mu
            else:
                inner_opts['threshold'] = opts['inner_tol']
            
            w_temp, inner_iter_nums[k * inner_iter + j] = fista(grad, L, lam, w_old, inner_opts)    
            w_new = w_old + l_r * (w_temp - w_old)
            w_old = w_new.copy()
            time_list.append(time()-tic)
        w_hat = w_new.copy()
        center_grad = f_svm_grad(w_hat, X, y, mu)
    return loss_sequence, w_sequence, error_sequence, inner_iter_nums, time_list[:-1]

def psvrg_svm(X, y, b=10, w_init=None, lr=1e-2, lam=1e-3, mu=1e-3, inner_iter=10, outer_iter=10):
    '''Stochastic Variance Reduced Gradient method
    '''
    n, d = X.shape
    w = w_init.copy()
    w_hat = w.copy()

    n_epochs = inner_iter * outer_iter
    loss = np.zeros(n_epochs)
    w_sequence = np.zeros((n_epochs, d))
    error_sequence = np.zeros(n_epochs)

    center_grad = f_svm_grad(w_hat, X, y, mu)

    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            p = np.random.permutation(n)
            loss[k * inner_iter + j] = F_svm(w, X, y, lam, mu)
            w_sequence[k * inner_iter + j] = w.copy()
            error_sequence[k * inner_iter + j] = np.linalg.norm(w - prox_h(w-f_svm_grad(w, X, y, mu), lam))

            grad = f_svm_grad(w, X[p[:b]], y[p[:b]], mu) - f_svm_grad(w_hat, X[p[:b]], y[p[:b]], mu) + center_grad
            w = prox_h(w - lr * grad, lr * lam)
            time_list.append(time()-tic)

        w_hat = w.copy()
        center_grad = f_svm_grad(w_hat, X, y, mu)
    return loss, w_sequence, error_sequence, time_list[:-1]