import numpy as np
from scipy.special import expit
# from scipy.sparse import multiply as sp_multiply
from scipy.sparse import csr_matrix

def h(w, lam=1e-3):  # L1-regularization
    return lam * np.linalg.norm(w, ord=1)


def prox_h(w, lam=0.01):  # proximal operator of lam * L1_norm
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)

'''l1-l2 regularized logistic regression'''
def f_logistic(w, X, y, mu=0): 
    '''logistic loss with \mu l_2 regularization
        w: parameter (1-dim array)
        X: feature matrix (2-dim array)
        y: labels (1-dim array) in {0, 1}
        mu: l_2 regularization parameter
    '''
    n = X.shape[0]
    z = X.dot(w)
    sigmoid = expit(z)
    loss = -y * np.log(sigmoid) - (1 - y) * np.log1p(-sigmoid)
    result = np.sum(loss) / n + mu / 2 * np.linalg.norm(w, ord=2)**2
    return result

def F_logistic(w, X, y, lam=1e-3, mu=0):  # total loss
    n = X.shape[0]
    z = X.dot(w)
    sigmoid = expit(z)
    loss = -y * np.log(sigmoid) - (1 - y) * np.log1p(-sigmoid)
    total_loss = np.sum(loss) / n + mu / 2 * np.linalg.norm(w, ord=2)**2 + lam * np.linalg.norm(w, ord=1)
    return total_loss

def f_logistic_grad(w, X, y, mu=0): 
    '''batch gradient of smooth part f
        X: batch feature matrix (2-dim array)
    '''
    n = X.shape[0]
    z = X.dot(w)
    c = expit(z)

    # Calculate the gradient
    error = c - y
    grad = X.T.dot(error) / n + mu * w
 
    return grad

def f_logistic_hess_z(w, X, z, mu=0):
    '''batch hessian vector product
        X: batch feature matrix (2-dim array)
        z: vector to be multiplied (1-dim array)
    '''
    n = X.shape[0]
    v = X.dot(w)
    c = expit(v)
    sigmoid_derivative = c * (1 - c)
    X_dot_z = X.dot(z)
    if isinstance(X, np.ndarray):
        hess_product = sigmoid_derivative[:, np.newaxis] * X * X_dot_z[:, np.newaxis]
        hess = hess_product.sum(axis=0)/n + mu * z
    else: # csr_sparse matrix
        hess_product = X.multiply(X_dot_z[:, np.newaxis]).multiply(sigmoid_derivative[:, np.newaxis]) # return a coo_matrix
        hess = hess_product.sum(axis=0)/n # return an array
        hess = np.asarray(hess).reshape(-1) + mu * z
    return hess

# nonconvex SVM

# def f_svm_old(w, X, y, mu=0): 
#     '''nonconvex SVM loss with \mu l_2 regularization
#         w: parameter (1-dim array)
#         X: feature matrix (2-dim array)
#         y: labels (1-dim array) in {-1, 1}
#         mu: l_2 regularization parameter
#     '''
#     n = X.shape[0]
#     z = X.dot(w)
#     sigmoid = expit(-2*y*z)
#     result = 2 * np.sum(sigmoid) / n + mu / 2 * np.linalg.norm(w, ord=2)**2
#     return result

def f_svm(w, X, y, mu=0):
    '''batch gradient of smooth part f
        X: batch feature matrix (2-dim array)
        y: labels (1-dim array) in {-1, 1} (not {0, 1})!!!
    '''
    n = X.shape[0]
    result = 1 + mu / 2 * np.linalg.norm(w, ord=2)**2 - np.sum(np.tanh(y * X.dot(w))) / n
    return result

# def f_svm_grad_old(w, X, y, mu=0):
#     '''batch gradient of smooth part f
#         X: batch feature matrix (2-dim array)
#         y: labels (1-dim array) in {-1, 1} (not {0, 1})!!!
#         It is L-smooth with L = 2 \sum\|x_i\|/n
#     '''
#     n = X.shape[0]
#     if isinstance(X, np.ndarray):
#         grad = np.sum(-y[:, np.newaxis] * X + y[:, np.newaxis] * X * np.tanh(y[:, np.newaxis] * np.dot(X, w)[:, np.newaxis]) ** 2, axis=0) / n + mu * w
#     else: # csr_sparse matrix
#         y = y[:, np.newaxis]
#         temp1 = -X.multiply(y) + X.multiply(np.tanh(np.multiply(y, X.dot(w))) ** 2).multiply(y)
#         grad = temp1.sum(axis=0) /n
#         grad = np.asarray(grad).reshape(-1) + mu * w
#     return grad

def f_svm_grad(w, X, y, mu=0):
    '''batch gradient of smooth part f
    X: batch feature matrix (2-dim array)
    '''
    n = X.shape[0]
    tanh_w = np.tanh(X.dot(w) * y) # 1-dim array (n,)
    X_y = X.T.dot(y) # 1-dim array (d,)
    Xy_tanh2 = X.T.dot(y*tanh_w**2) # 1-dim array (d,)
    grad = (-X_y + Xy_tanh2)/n + mu * w
    return grad

# def f_svm_hess_z(w, X, y, z, mu=0):
#     '''batch hessian vector product
#         X: batch feature matrix (2-dim array)
#         z: vector to be multiplied (1-dim array)
#     '''
#     n = X.shape[0]
#     x_dot_z = X.dot(z)
#     tanh_w = np.tanh(y * X.dot(w))
    
#     # Reshape the arrays to align for element-wise multiplication
#     tanh_w = tanh_w.reshape(-1, 1)
    
#     hessz = np.sum(2 * y[:, np.newaxis] * y[:, np.newaxis] * tanh_w * (1 - tanh_w ** 2) * x_dot_z * X, axis=0) / n + mu * z
#     return hessz

def f_svm_hess_z(w, X, y, z, mu=0):
    '''batch hessian vector product
        X: batch feature matrix (2-dim array)
        z: vector to be multiplied (1-dim array)
    '''
    n = X.shape[0]
    x_z = X.dot(z)
    tanh_w = np.tanh(X.dot(w) * y)
    tanh_w2 = tanh_w ** 2

    vec = 2 * (y ** 2) * tanh_w * (1 - tanh_w2) * x_z

    hessz = X.T.dot(vec) / n + mu * z
    
    return hessz

# def f_svm_hess_z_old(w, X, y, z, mu=0):
#     '''batch hessian vector product
#         X: batch feature matrix (2-dim array)
#         z: vector to be multiplied (1-dim array)
#     '''
#     n, d = X.shape
#     hessz = np.zeros(d)

#     for i in range(n):
#         x = X[i]
#         x_dot_z = x.dot(z)
#         tanh_w = np.tanh(y[i] * x.dot(w))
#         tanh_w2 = tanh_w ** 2

#         vec = 2 * (y[i] ** 2) * tanh_w * (1 - tanh_w2) * x_dot_z * x

#         hessz += vec
#     return hessz /n + mu * z

def F_svm(w, X, y, lam=1e-3, mu=0):  # total loss
    return f_svm(w, X, y, mu) + lam * np.linalg.norm(w, ord=1)