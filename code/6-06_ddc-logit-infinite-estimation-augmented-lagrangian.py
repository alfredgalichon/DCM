# Estimation of dynamic discrete choice via agumented Lagrangian
#
# Importing libraries and generating data:
import numpy as np
from scipy import optimize, sparse as sp
np.random.seed(7)
X,Y,K = 10,2,3
beta = 0.9
rel_tol = 1e-6
epsilon = 0.02
max_it = 1e6
phi_x_y_k = np.random.normal(size=(X,Y,K))/10
muhat_xy = np.random.uniform(size=(X*Y))
P_xprime_xy = np.random.uniform(size= (X,X*Y))
P_xprime_xy = P_xprime_xy / P_xprime_xy.sum(axis=0)[None,:]
# Computing:
phi_xy_k = phi_x_y_k.reshape((-1,K))
Psi = beta* P_xprime_xy.T - np.kron( np.eye(X),np.ones((Y,1)))
phi_infty = np.abs(phi_x_y_k).max() * muhat_xy.sum()
psi_infty = np.abs(Psi).max() * muhat_xy.sum()
SigmaY = np.kron( np.eye(X),np.ones((1,Y)))

def log_likelihood(lambda_and_u):
    lambda_k, u_x = lambda_and_u[:K], lambda_and_u[K:]
    return - muhat_xy.dot ( phi_xy_k @ lambda_k + Psi @ u_x )

def grad_log_likelihood(lambda_and_u):
    lambda_k, u_x = lambda_and_u[:K], lambda_and_u[K:]
    return - np.concatenate( (muhat_xy.dot( phi_xy_k), muhat_xy.dot( Psi)))

def Z_x(lambda_and_u):
    lambda_k, u_x = lambda_and_u[:K], lambda_and_u[K:]
    pi_x_y =  np.exp( phi_xy_k @ lambda_k + Psi @ u_x ).reshape((X,Y)) 
    return (pi_x_y.sum(axis=1)-1 )

def JZ_x(lambda_and_u):
    lambda_k, u_x = lambda_and_u[:K], lambda_and_u[K:]
    pi_xy =  np.exp( phi_xy_k @ lambda_k + Psi @ u_x )
    return  SigmaY @ np.diag(pi_xy) @ np.block([[phi_xy_k, Psi]])
    

def augmented_lagrangian(x, n_x, gamma, obj_val, obj_grad, constr_fun, constr_grad):
    f = obj_val(x)
    h = constr_fun(x)
    L = f + np.dot(n_x, h) + 0.5 * gamma * np.sum(h**2)
    grad_f = obj_grad(x)
    J_h = constr_grad(x)
    grad_L = grad_f + np.dot(J_h.T, n_x + gamma * h)
    return L, grad_L

def solve_augmented_lagrangian(x0, n0_x, gamma0, obj_val, obj_grad, constr_fun, constr_grad, coeff=10, max_iter=100, tol=1e-6):
    x = np.array(x0)
    n_x = np.array(n0_x)
    gamma = gamma0

    for i in range(max_iter):
        fx = lambda x: augmented_lagrangian(x, n_x, gamma, obj_val, obj_grad, constr_fun, constr_grad)[0]
        gx = lambda x: augmented_lagrangian(x, n_x, gamma, obj_val, obj_grad, constr_fun, constr_grad)[1]
        result = optimize.minimize(fx, x, method='BFGS', jac = gx)
        x = result.x
        h = constr_fun(x)
        n_x += gamma * h
        gamma *= coeff
        if np.linalg.norm(h, np.inf) < tol:
            break
    return x, n_x, gamma


lambda_and_u0 = np.zeros(K+X)
n0_x = muhat_xy.reshape((X,Y)).sum(axis=1)
gamma0 = 1.0

lambda_and_ustar, nstar_x, gammastar = solve_augmented_lagrangian(lambda_and_u0, n0_x, gamma0, log_likelihood, grad_log_likelihood, Z_x, JZ_x)

lambdastar_k, ustar_x = lambda_and_ustar[:K], lambda_and_ustar[K:]
print("lambda_k:", lambdastar_k)
print("n_x:", nstar_x)