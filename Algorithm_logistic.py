import numpy as np
from time import time
from tqdm import tqdm
from scipy.sparse.linalg import norm

from BFGS_class import BFGS
from problem_setup import f_logistic_grad, f_logistic_hess_z, F_logistic, prox_h, f_logistic
from fista import fista_backtracking, fista
from ssn_direct import ssn_direct
from CG_method import conjugate_gradient, conjugate_gradient_sub


def ISPQN_logistic(X, y, w_init=None, lr=1e-3, lam=1e-3, mu=1e-3, inner_iter=100, outer_iter=100, opts={}):
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
    opts.setdefault('Hessian', 'slbfgs') # stochastic L-BFGS in bryd's paper
    opts.setdefault('M', 10) # memory size
    opts.setdefault('r', 10) # Hessian update frequency
    opts.setdefault('b_H', 100) # batch size for Hessian update
    opts.setdefault('b', 100) # batch size
    opts.setdefault('window_size', 10)
    opts.setdefault('ssn_ratio', 1e-2)

    d = X.shape[1]
    if w_init is None:
        w_init = 0.1 * np.ones(d)
    if lam == 0:
        if opts['grad_scheme'] == 'svrg':
            return ISPQN_logistic_smooth(X, y, w_init, lr, mu, inner_iter, outer_iter, opts)
        elif opts['grad_scheme'] == 'sg':
            iter_num = inner_iter * outer_iter
            return SPQN_logistic_smooth(X, y, w_init, lr, mu, iter_num, opts)
        else:
            raise NotImplementedError('Gradient scheme {} is not implemented'.format(opts['grad_scheme']))
    else:
        if opts['grad_scheme'] == 'svrg':
            return subnewton_logistic_svrg_nonsmooth(X, y, w_init, lr, lam, mu, inner_iter, outer_iter, opts)
        elif opts['grad_scheme'] == 'sg':
            iter_num = inner_iter * outer_iter
            return subnewton_logistic_nonsmooth(X, y, w_init, lr, lam, mu, iter_num, opts)
        else:
            raise NotImplementedError('Gradient scheme {} is not implemented'.format(opts['grad_scheme']))

def ISPQN_logistic_smooth(X, y, w_init=None, learning_rate=1e-2, mu=1e-3, inner_iter=10, outer_iter=10, opts=None):
    '''subproblem is solved by CG'''
    M = opts['M']
    L = opts['r']
    b = opts['b']
    bH = opts['b_H']
    n, d = X.shape
    bfgs = BFGS(M)
    w_hat = w_init.copy() # reference point

    # initialize settings for slbfgs
    t = 0
    w_bar_old = np.zeros(d)
    w_bar_new = np.zeros(d)

    # initialize settings for algorithm
    w = w_init.copy()
    n_epochs = inner_iter * outer_iter
    loss = np.zeros(n_epochs)
    w_sequence = np.zeros((n_epochs, d))

    center_grad = f_logistic_grad(w_hat, X, y, mu)
    lr = learning_rate

    max_inner_iter = 0
    ave_inner_iter = 0
    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            p = np.random.permutation(n)

            # store current loss
            loss[k * inner_iter + j] = f_logistic(w, X, y, mu)
            w_sequence[k * inner_iter + j] = w.copy()

            # update correction pairs bfgs.s and bfgs.y
            if (k * inner_iter + j) % L == 0:
                t += 1
                w_bar_new /= L
                if t > 0:
                    S_H = p[:bH]  # indexes of selected points
                    s_new = w_bar_new - w_bar_old
                    y_new = f_logistic_hess_z(w_bar_new, X[p[S_H]], s_new, mu)
                    w_bar_old = w_bar_new.copy()
                    w_bar_new = np.zeros(d)

                    if np.dot(s_new, y_new) > 1e-15:
                        if bfgs.k_size < M:
                            bfgs.s.append(s_new)
                            bfgs.y.append(y_new)
                            bfgs.k_size += 1
                        else:
                            bfgs.s.popleft()
                            bfgs.y.popleft()
                            bfgs.s.append(s_new)
                            bfgs.y.append(y_new)

            # compute variance reduced stochastic gradient
            grad = f_logistic_grad(w, X[p[:b]], y[p[:b]], mu) - f_logistic_grad(w_hat, X[p[:b]], y[p[:b]], mu) + center_grad

            w_bar_new += w

            # update w
            if k * inner_iter + j < L:
                w = w - lr * grad

            else:
                # update bfgs information
                if bfgs.k_size == 0:
                    bfgs.sigma = 1
                else:
                    bfgs.sigma = np.dot(bfgs.y[-1], bfgs.y[-1]) / np.dot(bfgs.s[-1], bfgs.y[-1])      
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
        center_grad = f_logistic_grad(w_hat, X, y, mu)
    ave_inner_iter /= (outer_iter * inner_iter)
    return loss, w_sequence, [max_inner_iter, ave_inner_iter], time_list[:-1]
    
def ISPQN_logistic_nonsmooth(X, y, w_init=None, learning_rate=1e-2, lam=1e-3, mu=1e-3, inner_iter=10, outer_iter=10, opts=None):
    M = opts['M']
    L = opts['r']
    b = opts['b']
    bH = opts['b_H']
    n, d = X.shape
    bfgs = BFGS(M)
    w_hat = w_init.copy() # reference point

    # initialize settings for slbfgs
    t = 0
    w_bar_old = np.zeros(d)
    w_bar_new = np.zeros(d)

    # initialize settings for algorithm
    w = w_init.copy()
    n_epochs = inner_iter * outer_iter
    loss = np.zeros(n_epochs)
    w_sequence = np.zeros((n_epochs, d))
    center_grad = f_logistic_grad(w_hat, X, y, mu)
    lr = learning_rate

    acc_time = 0
    max_inner_iter = 0
    max_ave_search_len = 0
    ave_inner_iter = 0
    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            p = np.random.permutation(n)

            # store current loss
            loss[k * inner_iter + j] = F_logistic(w, X, y, lam, mu)
            w_sequence[k * inner_iter + j] = w.copy()

            # update correction pairs bfgs.s and bfgs.y
            if (k * inner_iter + j) % L == 0:
                t += 1
                w_bar_new /= L
                if t > 0:
                    S_H = p[:bH]  # indexes of selected points
                    s_new = w_bar_new - w_bar_old
                    y_new = f_logistic_hess_z(w_bar_new, X[p[S_H]], s_new, mu)
                    w_bar_old = w_bar_new.copy()
                    w_bar_new = np.zeros(d)

                    if np.dot(s_new, y_new) > 1e-15:
                        if bfgs.k_size < M:
                            bfgs.s.append(s_new)
                            bfgs.y.append(y_new)
                            bfgs.k_size += 1
                        else:
                            bfgs.s.popleft()
                            bfgs.y.popleft()
                            bfgs.s.append(s_new)
                            bfgs.y.append(y_new)

            # compute variance reduced stochastic gradient
            grad = f_logistic_grad(w, X[p[:b]], y[p[:b]], mu) - f_logistic_grad(w_hat, X[p[:b]], y[p[:b]], mu) + center_grad

            w_bar_new += w

            # update w
            if k * inner_iter + j < L:
                z = w - lr * grad
                w = prox_h(z, lr * lam)

            else:
                # update bfgs information
                if bfgs.k_size == 0:
                    bfgs.sigma = 1
                else:
                    bfgs.sigma = np.dot(bfgs.y[-1], bfgs.y[-1]) / np.dot(bfgs.s[-1], bfgs.y[-1])

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
                    w, iter = fista_backtracking(grad, bfgs, w, lr, lam, tol=opts['inner_tol'], L0=1, max_iter=opts['inner_max'], stopping=opts['stopping'])
                    max_inner_iter = max(max_inner_iter, iter)
                    ave_inner_iter += iter
                elif opts['inner_solver'] == 'ssn':
                    w, info = ssn_direct(grad, bfgs, w, lr, lam, mu, max_iter=opts['inner_max'], tol=opts['inner_tol'], stopping=opts['stopping'], ratio=opts['ssn_ratio']) # solve subproblem
                    acc_time += info[0]
                    max_inner_iter = max(max_inner_iter, info[1])
                    max_ave_search_len = max(max_ave_search_len, info[2])
                    ssn_info = [acc_time, max_inner_iter, max_ave_search_len]
                else:
                    raise NotImplementedError('optimizer {} is not implemented'.format(opts['inner_solver']))
            time_list.append(time()-tic)
        w_hat = w.copy()
        center_grad = f_logistic_grad(w_hat, X, y, mu)
    ave_inner_iter /= (outer_iter * inner_iter)
    if opts['inner_solver'] == 'ssn':
        return loss, w_sequence, ssn_info, time_list[:-1]
    else: 
        return loss, w_sequence, [max_inner_iter, ave_inner_iter], time_list[:-1]

def subnewton_logistic_svrg_nonsmooth(X, y, w_init=None, lr=1e-3, lam=1e-3, mu=1e-3, inner_iter=10, outer_iter=10, opts=None):
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
    center_grad = f_logistic_grad(w_hat, X, y, mu)

    # store sequences
    iter_num = outer_iter * inner_iter
    w_sequence = np.zeros((iter_num, d)) # store x_k
    loss_sequence = np.zeros(iter_num) # store F(x_k)
    inner_iter_nums = np.zeros(iter_num) # store number of inner iterations to update x_k

    flag_dense = isinstance(X, np.ndarray)

    inner_opts = {'stopping': opts['stopping'], 'max_iter': opts['inner_max']}
    time_list = [0]


    tic=time()

    # main loop
    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            w_sequence[k * inner_iter + j] = w_old
            loss_sequence[k * inner_iter + j] = F_logistic(w_old, X, y, lam, mu)

            # select independent samples O_k and S_k
            p1 = np.random.permutation(n)
            p2 = np.random.permutation(n)
            O = p1[:b]
            S = p2[:bH]

            # set up for inner FISTA solver
            g_k = f_logistic_grad(w_old, X[O], y[O], mu) - f_logistic_grad(w_hat, X[O], y[O], mu) + center_grad

            # use np.linalg.norm() for narray and norm() for sparse matrix (very important!!!)
            if flag_dense:
                L = np.max(np.linalg.norm(X[S], axis=1, ord=2)**2) + mu
            else: # csr_sparse matrix
                L = np.max(norm(X[S], axis=1, ord=2)**2) + mu
            def grad(w): # the gradient of inner FISTA, linear in w
                return g_k + f_logistic_hess_z(w_old, X[S], w-w_old, mu)
            
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
        center_grad = f_logistic_grad(w_hat, X, y, mu)
    return loss_sequence, w_sequence, inner_iter_nums, time_list[:-1]
    
def SPQN_logistic_smooth(X, y, w_init=None, learning_rate=1e-2, mu=1e-3, outer_iter=100, opts=None):
    '''subproblem is solved by CG'''
    M = opts['M']
    L = opts['r']
    b = opts['b']
    bH = opts['b_H']
    n, d = X.shape
    bfgs = BFGS(M)

    window_size = opts['window_size']

    # initialize settings for slbfgs
    t = 0
    w_bar_old = np.zeros(d)
    w_bar_new = np.zeros(d)

    # initialize settings for algorithm
    w = w_init.copy()
    loss = np.zeros(outer_iter)
    w_sequence = np.zeros((outer_iter, d))

    lr = learning_rate

    max_inner_iter = 0
    ave_inner_iter = 0
    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        if k % window_size == 0:
            lr /= 2
        p = np.random.permutation(n)

        # store current loss
        loss[k] = f_logistic(w, X, y, mu)
        w_sequence[k] = w.copy()

        # update correction pairs bfgs.s and bfgs.y
        if k % L == 0:
            t += 1
            w_bar_new /= L
            if t > 0:
                S_H = p[:bH]  # indexes of selected points
                s_new = w_bar_new - w_bar_old
                y_new = f_logistic_hess_z(w_bar_new, X[p[S_H]], s_new, mu)
                w_bar_old = w_bar_new.copy()
                w_bar_new = np.zeros(d)

                if np.dot(s_new, y_new) > 1e-15:
                    if bfgs.k_size < M:
                        bfgs.s.append(s_new)
                        bfgs.y.append(y_new)
                        bfgs.k_size += 1
                    else:
                        bfgs.s.popleft()
                        bfgs.y.popleft()
                        bfgs.s.append(s_new)
                        bfgs.y.append(y_new)

        # compute batch stochastic gradient
        grad = f_logistic_grad(w, X[p[:b]], y[p[:b]], mu)

        w_bar_new += w

        # update w
        if k < L:
            w = w - lr * grad

        else:
            # update bfgs information
            if bfgs.k_size == 0:
                bfgs.sigma = 1
            else:
                bfgs.sigma = np.dot(bfgs.y[-1], bfgs.y[-1]) / np.dot(bfgs.s[-1], bfgs.y[-1])      
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
    ave_inner_iter /= outer_iter
    return loss, w_sequence, [max_inner_iter, ave_inner_iter], time_list[:-1]
    
def SPQN_logistic_nonsmooth(X, y, w_init=None, learning_rate=1e-2, lam=1e-3, mu=1e-3, outer_iter=100, opts=None):
    M = opts['M']
    L = opts['r']
    b = opts['b']
    bH = opts['b_H']
    n, d = X.shape
    bfgs = BFGS(M)

    window_size = opts['window_size']

    # initialize settings for slbfgs
    t = 0
    w_bar_old = np.zeros(d)
    w_bar_new = np.zeros(d)

    # initialize settings for algorithm
    w = w_init.copy()
    loss = np.zeros(outer_iter)
    w_sequence = np.zeros((outer_iter, d))
    lr = learning_rate

    acc_time = 0
    max_inner_iter = 0
    max_ave_search_len = 0
    ave_inner_iter = 0
    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        if k % window_size == 0:
            lr /= 2
        p = np.random.permutation(n)

        # store current loss
        loss[k] = F_logistic(w, X, y, lam, mu)
        w_sequence[k] = w.copy()

        # update correction pairs bfgs.s and bfgs.y
        if k % L == 0:
            t += 1
            w_bar_new /= L
            if t > 0:
                S_H = p[:bH]  # indexes of selected points
                s_new = w_bar_new - w_bar_old
                y_new = f_logistic_hess_z(w_bar_new, X[p[S_H]], s_new, mu)
                w_bar_old = w_bar_new.copy()
                w_bar_new = np.zeros(d)

                if np.dot(s_new, y_new) > 1e-15:
                    if bfgs.k_size < M:
                        bfgs.s.append(s_new)
                        bfgs.y.append(y_new)
                        bfgs.k_size += 1
                    else:
                        bfgs.s.popleft()
                        bfgs.y.popleft()
                        bfgs.s.append(s_new)
                        bfgs.y.append(y_new)

        # compute batch stochastic gradient
        grad = f_logistic_grad(w, X[p[:b]], y[p[:b]], mu)

        w_bar_new += w

        # update w
        if k < L:
            z = w - lr * grad
            w = prox_h(z, lr * lam)

        else:
            # update bfgs information
            if bfgs.k_size == 0:
                bfgs.sigma = 1
            else:
                bfgs.sigma = np.dot(bfgs.y[-1], bfgs.y[-1]) / np.dot(bfgs.s[-1], bfgs.y[-1])

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
                w, iter = fista_backtracking(grad, bfgs, w, lr, lam, tol=opts['inner_tol'], L0=1, max_iter=opts['inner_max'], stopping=opts['stopping'])
                max_inner_iter = max(max_inner_iter, iter)
                ave_inner_iter += iter
            elif opts['inner_solver'] == 'ssn':
                w, info = ssn_direct(grad, bfgs, w, lr, lam, mu, max_iter=opts['inner_max'], tol=opts['inner_tol'], stopping=opts['stopping'], ratio=opts['ssn_ratio']) # solve subproblem
                acc_time += info[0]
                max_inner_iter = max(max_inner_iter, info[1])
                max_ave_search_len = max(max_ave_search_len, info[2])
                ssn_info = [acc_time, max_inner_iter, max_ave_search_len]
            else:
                raise NotImplementedError('optimizer {} is not implemented'.format(opts['inner_solver']))
        time_list.append(time()-tic)
    ave_inner_iter /= outer_iter
    if opts['inner_solver'] == 'ssn':
        return loss, w_sequence, ssn_info, time_list[:-1]
    else: 
        return loss, w_sequence, [max_inner_iter, ave_inner_iter], time_list[:-1]
    
def subnewton_logistic_nonsmooth(X, y, w_init=None, lr=1e-3, lam=1e-3, mu=1e-3, iter_num=100, opts=None):
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

    # store sequences
    w_sequence = np.zeros((iter_num, d)) # store x_k
    loss_sequence = np.zeros(iter_num) # store F(x_k)
    inner_iter_nums = np.zeros(iter_num) # store number of inner iterations to update x_k

    flag_dense = isinstance(X, np.ndarray)

    inner_opts = {'stopping': opts['stopping'], 'max_iter': opts['inner_max']}
    time_list = [0]


    tic=time()

    # main loop
    for k in tqdm(range(iter_num)):
        w_sequence[k] = w_old
        loss_sequence[k] = F_logistic(w_old, X, y, lam, mu)

        # select independent samples O_k and S_k
        p1 = np.random.permutation(n)
        p2 = np.random.permutation(n)
        O = p1[:b]
        S = p2[:bH]

        # set up for inner FISTA solver
        g_k = f_logistic_grad(w_old, X[O], y[O], mu)

        # use np.linalg.norm() for narray and norm() for sparse matrix (very important!!!)
        if flag_dense:
            L = np.max(np.linalg.norm(X[S], axis=1, ord=2)**2) + mu
        else: # csr_sparse matrix
            L = np.max(norm(X[S], axis=1, ord=2)**2) + mu
        def grad(w): # the gradient of inner FISTA, linear in w
            return g_k + f_logistic_hess_z(w_old, X[S], w-w_old, mu)
        
        # define options for inner FISTA
        if inner_opts['stopping'] == 'our':
            inner_opts['threshold'] = opts['inner_tol'] * mu
        else:
            inner_opts['threshold'] = opts['inner_tol']
        
        w_temp, inner_iter_nums[k] = fista(grad, L, lam, w_old, inner_opts)    
        w_new = w_old + l_r * (w_temp - w_old)
        w_old = w_new.copy()
        time_list.append(time()-tic)
    return loss_sequence, w_sequence, inner_iter_nums, time_list[:-1]
    
def psvrg_logistic(X, y, b=10, w_init=None, lr=1e-2, lam=1e-3, mu=1e-3, inner_iter=10, outer_iter=10):
    '''Stochastic Variance Reduced Gradient method (including smooth and nonsmooth cases)
    '''
    n, d = X.shape
    w = w_init.copy()
    w_hat = w.copy()

    n_epochs = inner_iter * outer_iter
    loss = np.zeros(n_epochs)
    w_sequence = np.zeros((n_epochs, d))
    center_grad = f_logistic_grad(w_hat, X, y, mu)

    time_list = [0]
    tic=time()

    for k in tqdm(range(outer_iter)):
        for j in range(inner_iter):
            p = np.random.permutation(n)
            loss[k * inner_iter + j] = F_logistic(w, X, y, lam, mu)
            w_sequence[k * inner_iter + j] = w.copy()
            grad = f_logistic_grad(w, X[p[:b]], y[p[:b]], mu) - f_logistic_grad(w_hat, X[p[:b]], y[p[:b]], mu) + center_grad
            w = prox_h(w - lr * grad, lr * lam)
            time_list.append(time()-tic)

        w_hat = w.copy()
        center_grad = f_logistic_grad(w_hat, X, y, mu)
    return loss, w_sequence, time_list[:-1]

def sub_newton_cg(X, y, w_init=None, lr=1e-2, mu=1e-3, iter_num=100, opts=None):
    '''the subsampled Newton method with CG solver in the work [Exact and Inexact Subsampled Newton Methods for
 Optimization, 2018]'''
    n, d = X.shape
    b = opts['b']
    b_H = opts['b_H']
    if w_init is None:
        w_init = 0.01 * np.ones(d)
    w = w_init.copy()
    loss = np.zeros(iter_num)
    w_sequence = np.zeros((iter_num, d))
    max_inner_iter = 0
    ave_inner_iter = 0
    time_list = [0]
    tic=time()

    for k in tqdm(range(iter_num)):
        if k % opts['window_size'] == 0:
            lr /= 2
        p = np.random.permutation(n)
        loss[k] = f_logistic(w, X, y, mu)
        w_sequence[k] = w.copy()
        grad = f_logistic_grad(w, X[p[:b]], y[p[:b]], mu)
        d, iter = conjugate_gradient_sub(X, mu, grad, p[:b_H], w, opts['inner_max'], opts['inner_tol'], stopping=opts['stopping'])
        w = w + lr * d
        max_inner_iter = max(max_inner_iter, iter)
        ave_inner_iter += iter
        time_list.append(time()-tic)
    ave_inner_iter /= iter_num
    return loss, w_sequence, [max_inner_iter, ave_inner_iter], time_list[:-1]