# Estimation of dynamic discrete choice via nested fixed point
#
# Importing libraries and generating data:
import numpy as np
from scipy import optimize, sparse as sp
np.random.seed(7)
X,Y,K = 10,2,3
beta = 0.9
rel_tol = 1e-5
epsilon = 0.05
max_it = 10000
phi_x_y_k = np.random.normal(size=(X,Y,K))/10
muhat_xy = np.random.uniform(size=(X*Y))
P_xprime_xy = np.random.uniform(size= (X,X*Y))
P_xprime_xy = P_xprime_xy / P_xprime_xy.sum(axis=0)[None,:]
# Computing:
phi_xy_k = phi_x_y_k.reshape((-1,K))
SigmaY = sp.kron( sp.eye(X),np.ones((1,Y)))
Psi = beta* P_xprime_xy.T - np.kron( np.eye(X),np.ones((Y,1)))
tol_lambda = rel_tol* np.abs(phi_x_y_k).max()*muhat_xy.sum()

def fu_x(lambda_k, tol_u=1e-5, max_it = 10000):
    u_x = np.zeros(X)
    phi_x_y = (phi_xy_k @ lambda_k).reshape((X,Y))
    M = 1+ (np.abs(phi_x_y).max() + np.log(Y)) / (1-beta)
    S = int(max ( (np.log(tol_u) - 2* M) / np.log(beta) ,10) )
    for s in range(min(S,max_it)):
        m_x = (phi_x_y + beta * (P_xprime_xy.T @ u_x).reshape((X,Y))).max(axis = 1)
        u_x = m_x + np.log( np.exp(phi_x_y + beta * (P_xprime_xy.T @ u_x).reshape((X,Y)) -m_x[:,None] ).sum(axis = 1) )
    return(u_x)

def fdu_x(lambda_k):
    u_x = fu_x(lambda_k).flatten()
    pi_xy = np.exp( phi_xy_k @ lambda_k + Psi @ u_x)
    return - sp.linalg.spsolve(SigmaY @ sp.diags(pi_xy) @ Psi , SigmaY @ sp.diags(pi_xy) @ phi_xy_k)

def neg_F(lambda_k):
    return  - muhat_xy.T @ ( phi_xy_k @ lambda_k + Psi @ fu_x(lambda_k))

def neg_gradF(lambda_k):
    return muhat_xy.T @ ( - Psi @ fdu_x(lambda_k) - phi_xy_k)

def neg_hessF(lambda_k):
    u_x = fu_x(lambda_k).flatten()
    pi_xy = np.exp( phi_xy_k @ lambda_k + Psi @ u_x)
    dlogpi = phi_xy_k + Psi @ fdu_x(lambda_k)
    XY = X*Y
    TTT = sp.csr_matrix( ([1]*XY,(list(range(XY)), [a*(XY+1) for a in range(XY)])),shape = (XY,XY*XY) )
    Tdlogpisquare =  TTT @ sp.kron(dlogpi,dlogpi )
    d2u = - sp.linalg.spsolve( SigmaY @ sp.diags(pi_xy) @ Psi , SigmaY @ sp.diags(pi_xy) @  Tdlogpisquare)
    return (- muhat_xy.T @ Psi @ d2u).reshape((K,K))

result = optimize.minimize(neg_F, np.zeros(K), jac=neg_gradF, method='BFGS')

# displaying results
if result.success:
    lambdahat_k = result.x
    print("lambdahat:", lambdahat_k)
    I = muhat_xy.sum()
    V = np.diag(muhat_xy) / I - muhat_xy[:,None] @ muhat_xy[None,:] / (I*I)
    HinvB = np.linalg.solve(neg_hessF(lambdahat_k), (phi_xy_k + Psi @ fdu_x(lambdahat_k)).T)
    print('Cov(lambdahat)=')
    print(HinvB @ V @ HinvB.T)
else:
    print(result.message)