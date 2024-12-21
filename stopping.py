import numpy as np
from auxiliary_functions import B_alpha_z_direct
from problem_setup import prox_h


def check_stopping_nonsmooth(v_k, bfgs, x_k, x, eta, lam, t):
    ''' v_k: gradient estimation at x_k
        bfgs: BFGS object
        x_k: current point
        x: tested point
        eta: learning rate
        t: parameter in G(x)
    '''
    check = False
    G = (x - prox_h(x - t * v_k - t * B_alpha_z_direct(x-x_k, 0, bfgs) / eta, lam * t)) / t
    if np.linalg.norm(G) < np.linalg.norm(x-x_k) / eta:
        check = True
    return check

def check_stopping(grad, x, lam, eps):
    '''grad: gradient function of smooth part, take one parameter
       x: tested point (1-dim array)
       lam: coefficient of L1-norm
       eps: accuracy threshold    
    '''
    # initialization
    check = False
    d = len(x)
    g = grad(x)
    I = np.nonzero(x)
    Ic = np.where(x == 0)
    r = np.zeros_like(x) # residual
    if len(I) == d:
        r = g + lam * np.sign(x)
    else:
        r[I] = g[I] + lam * np.sign(x[I])   
        for i in Ic:
            r[i] = np.clip(0, -lam+g[i], lam+g[i])
    if np.linalg.norm(r) < eps:
        check = True
    return check