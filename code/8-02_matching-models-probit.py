# Estimation of a matching model with probit heterogeneity
#
# Importing libraries and generating data:
#
import numpy as np, scipy.sparse as sp, gurobipy as grb
X,Y,K = 5,4,3
I=J= 10 # with a license, take I=J=1000
np.random.seed(7)
phi_xy_k = np.random.uniform(size=(X*Y,K))
muhat_xy = np.random.uniform(size=(X*Y))
# Simulating heterogeneity:
epsilon_ixy = np.random.normal(size = (I*X*Y))
epsilon_ix0 = np.random.normal(size = I*X)
eta_jxy = np.random.normal(size = (J*X*Y))
eta_jy0 = np.random.normal(size = J*Y )
# Computing:
n_x = muhat_xy.reshape((X,Y)).sum(axis=1)
m_y = muhat_xy.reshape((X,Y)).sum(axis=0)
m=grb.Model()
lambda_k = m.addMVar(K, lb= -grb.GRB.INFINITY)
u_ix = m.addMVar(I*X, lb= -grb.GRB.INFINITY)
v_jy = m.addMVar(J*Y, lb= -grb.GRB.INFINITY)
U_xy =  m.addMVar(X*Y, lb= -grb.GRB.INFINITY)
V_xy =  m.addMVar(X*Y, lb= -grb.GRB.INFINITY)
#theta = m.addMVar(len(b), lb= -grb.GRB.INFINITY)
m.setObjective( (muhat_xy @ phi_xy_k)@ lambda_k -(u_ix.reshape((I,X))*n_x[None,:]).sum()/I 
               - (v_jy.reshape((I,Y))*m_y[None,:]).sum()/I ,sense=grb.GRB.MAXIMIZE)
m.addConstr(U_xy+V_xy  >= phi_xy_k @ lambda_k)
m.addConstr(u_ix.reshape((I,X))[:,:,None] >= U_xy.reshape((X,Y))[None,:,:] + epsilon_ixy.reshape((I,X,Y)) )
m.addConstr(u_ix >= epsilon_ix0)
m.addConstr(v_jy.reshape((J,Y))[:,None,:] >= V_xy.reshape((X,Y))[None,:,:] + eta_jxy.reshape((J,X,Y)) )
m.addConstr(v_jy >= eta_jy0)
m.optimize()
# displaying results:
lambda_k = np.array(m.x[:K])
print('lambda_k=',lambda_k)
