import numpy as np
from tqdm import tqdm
from time import time

from problem_setup import prox_h
from stopping import check_stopping_nonsmooth, check_stopping
from auxiliary_functions import B_alpha_z_direct


def fista(grad, L, lam, w_init, options=None):
    '''FISTA for solving a general minimization problem
        problem setup:
            grad: gradient function, take one parameter w
            L: smoothness constant of the smooth part f
            lam: coefficient of L1-norm
        algorithm setup:
            w_init: initial point, should be chosen as w_k!!!
            options: a dict containing keys 'max_iter', 'stopping' and 'threshold'    
    '''
    # initialization
    options.setdefault('max_iter', 1000)
    options.setdefault('stopping', '')
    options.setdefault('threshold', 1e-8)
    options.setdefault('store_seq', False)
    options.setdefault('time', False)
    
    max_iter = options['max_iter']
    d = len(w_init)
    iter_num = 0
    x_old = w_init.copy()
    y_old = w_init.copy()
    t_old = 1

    time_list = []
    tic = time()

    if options['store_seq']:
        x_sequence = np.zeros((max_iter, d))

    # main loop
    if options['stopping'] == 'our':
        while iter_num < options['max_iter']:
            if options['store_seq']:
                x_sequence[iter_num] = x_old
            x_new = prox_h(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)
            eps = options['threshold']*np.linalg.norm(x_new-w_init, ord=2)**2
            if check_stopping(grad, x_new, lam, eps):
                break
            x_old, y_old, t_old = x_new.copy(), y_new.copy(), t_new
            iter_num += 1
            time_list.append(time()-tic)
    else:
        def error(w): # first-order optimal condition 
            return w - prox_h(w - grad(w), lam)

        while iter_num < options['max_iter']:
            if options['store_seq']:
                x_sequence[iter_num] = x_old
            x_new = prox_h(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)

            e = np.linalg.norm(error(x_new)) / np.linalg.norm(error(w_init))
            if e < options['threshold']:
                break
            x_old, y_old, t_old = x_new.copy(), y_new.copy(), t_new
            iter_num += 1
            time_list.append(time()-tic)
    if options['store_seq']:
        if options['time']:
            return x_sequence, iter_num, time_list
        else:
            return x_sequence, iter_num
    else:
        if options['time']:
            return x_new, iter_num, time_list
        else:
            return x_new, iter_num
        
def fista_backtracking(v, bfgs, x_current, eta=1e-2, lam=1e-3, tol=1e-2, L0=1, max_iter=1000, stopping='our', opts={}):
    '''
    for solving subproblem in form of min <v, x-x_k> + <x-x_k, H_k(x-x_k)>/(2 \eta) + h(x)
    v: gradient estimation at x_k
    bfgs: BFGS object
    x_current: current point x_k
    eta: learning rate
    tol: t in the function check_stopping_nonsmooth(), proportional to eta
    L0: initial Lipschitz constant
    '''
    opts.setdefault('line_search_length', 100)
    opts.setdefault('rho', 2)
    def Q(x, y, L): # Q function after simplification, note that f(x) = <v, x-x_k> + <x-x_k, H_k(x-x_k)>/(2 \eta)
        return v.dot(x-x_current) +(2*x - y - x_current).dot(B_alpha_z_direct(y-x_current, 0, bfgs))/(2*eta) + (L / 2) * np.linalg.norm(x - y) ** 2 + lam*np.linalg.norm(x, ord=1)

    def F(x):
        return v.dot(x-x_current) + (x-x_current).dot(B_alpha_z_direct(x-x_current, 0, bfgs))/(2*eta) + lam*np.linalg.norm(x, ord=1)
    
    # initialization
    x_init = x_current.copy()
    x_old = x_init.copy()
    y_old = x_init.copy()
    t_old = 1
    iter = 0
    L = L0

    error_0 = y_old - prox_h(y_old - (v+B_alpha_z_direct(y_old-x_current, 0, bfgs)/eta), lam)
    def grad(x):
        return v + B_alpha_z_direct(x - x_current, 0, bfgs)/eta
    
    # flag for checking whether the desired $L$ is found
    flag = False 

    # main loop
    for iter in range(max_iter):
        # backtracking to find L
        Lbar = L
        x_trial = prox_h(y_old - (v+B_alpha_z_direct(y_old-x_current, 0, bfgs)/eta)/Lbar, lam / Lbar)
        count = 0
        while F(x_trial) > Q(x_trial, y_old, Lbar) and count < opts['line_search_length']:
            Lbar = Lbar * opts['rho']
            x_trial = prox_h(y_old - (v+B_alpha_z_direct(y_old-x_current, 0, bfgs)/eta)/Lbar, lam / Lbar)
            count += 1
        if count == opts['line_search_length']:
            flag = True
            break
        # update x, y, t
        L = Lbar
        x_new = x_trial.copy()
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old ** 2))
        y_new = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

        if stopping == 'our':
            if check_stopping(grad, y_new, lam, tol): # stopping criterion, tol is t in the definition of check_stopping_nonsmooth()
                break
        else:
            error = y_new - prox_h(y_new - (v+B_alpha_z_direct(y_new-x_current, 0, bfgs)/eta), lam)
            if np.linalg.norm(error) / np.linalg.norm(error_0) < tol:
                break
        x_old, y_old, t_old = x_new.copy(), y_new.copy(), t_new
    if flag:
        raise ValueError('Line search for determing L failed, please choose larger max_iter')
    else:
        return y_new, iter+1