# Estimation of a model of finite-horizon dynamic discrete choice  with probit heterogeneity
# Importing libraries and generating data:
!pip install gurobipy
import numpy as np, gurobipy as grb, scipy.sparse as sp
X,Y,T,K = 5,4,10,3
I=J= 3 # with a license, take I=J=1000
np.random.seed(7)
phi_txy_k = np.random.uniform(size=(T*X*Y,K))
muhat_txy = np.random.uniform(size=(T*X*Y))
muhat_txy = ( muhat_txy.reshape((T,X,Y)) / muhat_txy.reshape((T,X,Y)).sum(axis = (1,2) )[:,None,None]).flatten()
P_xprime_xy = np.random.uniform(size= (X,X*Y))
P_xprime_xy = P_xprime_xy / P_xprime_xy.sum(axis=0)[None,:]
JT = sp.diags([1], offsets=-1, shape=(T, T), format='csr')
# Simulating heterogeneity:
epsilon_itxy = np.random.normal(size = (I*T*X*Y))
# Computing:
m = grb.Model()
lambda_k = m.addMVar(K,lb = - grb.GRB.INFINITY)
u_tx = m.addMVar((T*X),lb = - grb.GRB.INFINITY)
utilde_itx = m.addMVar((I*T*X),lb = - grb.GRB.INFINITY)
Psi = sp.kron(JT.T,P_xprime_xy.T) -sp.kron(sp.eye(T),sp.kron(sp.eye(X),np.ones((Y,1)) ) )

m.setObjective(  ( (muhat_txy @ phi_txy_k) @ lambda_k  + (muhat_txy @Psi) @ u_tx ) ,sense = grb.GRB.MAXIMIZE)
m.addConstr(u_tx == sp.kron(np.ones((1,I))/I,sp.eye(T*X)) @ utilde_itx )
m.addConstr( sp.kron(sp.eye(I*T*X), np.ones((Y,1)))@ utilde_itx >= sp.kron( np.ones((I,1)),phi_txy_k ) @ lambda_k
           + sp.kron( np.ones((I,1)) ,sp.kron(JT.T, P_xprime_xy.T)) @ u_tx +  epsilon_itxy)
m.optimize()
# displaying results:
lambda_k = np.array(m.x[:K])
print('lambda_k=',lambda_k)