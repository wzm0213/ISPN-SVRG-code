import numpy as np
from auxiliary_functions import B_alpha_z_direct
from problem_setup import f_logistic_hess_z

# CG method
def conjugate_gradient(v_k, eta, bfgs, x_current, max_iter=1000, eps=1e-2, stopping='our'):
    '''solve Ax = b, with A:=H_k, b:=H_k x_k - \eta v_k
        Ax := B_alpha_z_direct(x, 0, bfgs)
    '''
    # Initialize
    b = B_alpha_z_direct(x_current, 0, bfgs) - eta * v_k
    x = x_current.copy()
    r_old = b-B_alpha_z_direct(x, 0, bfgs)
    if stopping == 'our':
        tol = eps * np.linalg.norm(v_k)
    else:
        tol = eps * np.linalg.norm(r_old)
    if np.linalg.norm(r_old) < tol:
        return x

    p = r_old.copy()

    # Main loop
    for iter in range(max_iter):
        # Compute alpha
        Ap = B_alpha_z_direct(p, 0, bfgs)
        alpha = np.linalg.norm(r_old)**2/np.dot(p, Ap)
        # Update x
        x = x + alpha * p
        # Compute new gradient
        r_new = r_old - alpha * Ap
        # Check stopping criterion
        if np.linalg.norm(r_new) < tol:
            break
        # Compute beta
        beta = np.linalg.norm(r_new)**2/np.linalg.norm(r_old)**2
        # Update p
        p = r_new + beta * p
        # Update r
        r_old = r_new.copy()
    return x, iter + 1

def conjugate_gradient_sub(X, mu, grad, hess_index, x_current, max_iter=1000, eps=1e-2, stopping='our'):
    '''solve Ax = b, with A:=\nabla^2 f_S(x_k), b:=-\nabla f_O(x_k)
    '''
    # Initialize
    b = -grad
    x = x_current.copy()
    r_old = b - f_logistic_hess_z(x_current, X[hess_index], x, mu) # b-Ax
    if stopping == 'our':
        tol = eps * np.linalg.norm(grad)
    else:
        tol = eps * np.linalg.norm(r_old)
    if np.linalg.norm(r_old) < tol:
        return x

    p = r_old.copy()

    # Main loop
    for iter in range(max_iter):
        # Compute alpha
        Ap = f_logistic_hess_z(x_current, X[hess_index], p, mu)
        alpha = np.linalg.norm(r_old)**2/np.dot(p, Ap)
        # Update x
        x = x + alpha * p
        # Compute new gradient
        r_new = r_old - alpha * Ap
        # Check stopping criterion
        if np.linalg.norm(r_new) < tol:
            break
        # Compute beta
        beta = np.linalg.norm(r_new)**2/np.linalg.norm(r_old)**2
        # Update p
        p = r_new + beta * p
        # Update r
        r_old = r_new.copy()
    return x, iter + 1

def conjugate_gradient_matrix(A, b, x_current, max_iter=1000, theta=1e-2):
    '''solve Ax = b, for testing purpose
    '''
    # Initialize
    x = x_current.copy()
    r_old = b-A.dot(x)
    eps = theta

    if np.linalg.norm(r_old) < eps:
        return x

    p = r_old.copy()

    # Main loop
    for i in range(max_iter):
        # Compute alpha
        Ap = A.dot(p)
        alpha = np.linalg.norm(r_old)**2/np.dot(p, Ap)
        # Update x
        x = x + alpha * p
        # Compute new gradient
        r_new = r_old - alpha * Ap
        # Check stopping criterion
        if np.linalg.norm(r_new) < eps:
            break
        # Compute beta
        beta = np.linalg.norm(r_new)**2/np.linalg.norm(r_old)**2
        # Update p
        p = r_new + beta * p
        # Update r
        r_old = r_new.copy()
    return x