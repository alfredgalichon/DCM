# Estimation of dynamic discrete choice via scipy.optimize
#
# Importing libraries and generating data:
import numpy as np
from scipy import optimize, sparse as sp
np.random.seed(7)
T,X,Y,K = 2,3,2,3
beta = 0.9
phi_x_y_k = np.random.normal(size=(X,Y,K))
muhat_txy = np.random.uniform(size=(T*X*Y))
P_xprime_xy = np.random.uniform(size= (X,X*Y))
P_xprime_xy = P_xprime_xy / P_xprime_xy.sum(axis=0)[None,:]
# Computing:
phi_t_x_y_k = np.array([beta**t for t in range(T)])[:,None,None,None] *  phi_x_y_k[None,:,:,:]
phi_txy_k = phi_t_x_y_k.reshape((-1,K))
SigmaY = sp.kron( sp.eye(T*X),np.ones((1,Y)))
Psi = sp.kron(sp.diags(diagonals=[1]*(T-1), offsets=+1, shape=(T,T)), P_xprime_xy.T) - SigmaY.T

def fu_tx(lambda_k):
  u_t_x = np.zeros((T+1,X))
  for t in range(T-1,-1,-1):
    u_t_x[t,:] = np.log( np.exp( phi_t_x_y_k[t,:,:,:].reshape((-1,K)) @ lambda_k +(u_t_x[t+1,:] @ P_xprime_xy)).reshape((X,Y)).sum(axis = 1))
  return u_t_x[:-1,:].flatten()

def fdu_tx(lambda_k):
    u_tx = fu_tx(lambda_k).flatten()
    pi_txy = np.exp( phi_txy_k @ lambda_k + Psi @ u_tx)
    return - sp.linalg.spsolve(SigmaY @ sp.diags(pi_txy) @ Psi , SigmaY @ sp.diags(pi_txy) @ phi_txy_k)

def neg_F(lambda_k):
    return  - muhat_txy.T @ ( phi_txy_k @ lambda_k + Psi @ fu_tx(lambda_k))

def neg_gradF(lambda_k):
    return muhat_txy.T @ ( - Psi @ fdu_tx(lambda_k) - phi_txy_k)

def neg_hessF(lambda_k):
    u_tx = fu_tx(lambda_k).flatten()
    pi_txy = np.exp( phi_txy_k @ lambda_k + Psi @ u_tx)
    dlogpi = phi_txy_k + Psi @ fdu_tx(lambda_k)
    TXY = T*X*Y
    TTT = sp.csr_matrix( ([1]*TXY,(list(range(TXY)), [a*(TXY+1) for a in range(TXY)])),shape = (TXY,TXY*TXY) )
    Tdlogpisquare =  TTT @ sp.kron(dlogpi,dlogpi )
    d2u = - sp.linalg.spsolve( SigmaY @ sp.diags(pi_txy) @ Psi , SigmaY @ sp.diags(pi_txy) @  Tdlogpisquare)
    return (- muhat_txy.T @ Psi @ d2u).reshape((K,K))

result = optimize.minimize(neg_F, np.zeros(K), jac=neg_gradF, method='BFGS')

# displaying results
if result.success:
    lambdahat_k = result.x
    print("lambdahat:", lambdahat_k)
    I = muhat_txy.sum()
    V= np.diag(muhat_txy) / I - muhat_txy[:,None] @ muhat_txy[None,:] / (I*I)
    HinvB = np.linalg.solve(neg_hessF(lambdahat_k), (phi_txy_k + Psi @ fdu_tx(lambdahat_k)).T)
    print('Cov(lambdahat)=')
    print(HinvB @ V @ HinvB.T)
else:
    print(result.message)