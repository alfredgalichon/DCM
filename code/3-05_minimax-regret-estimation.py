# Minimax-regret estimation
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp, gurobipy as grb
np.random.seed(7)
I,Y,K = 500,4,3
muhat_iy = np.random.multinomial(1, [1/Y]*Y, size=I).flatten()
phi_iy_k = np.random.normal( size = (I*Y,K))
phi_iy0 = np.random.normal( size = I*Y )
# Computing:
m=grb.Model()
u_i = m.addMVar(I,lb = - grb.GRB.INFINITY)
lambda_k = m.addMVar(K, lb = - grb.GRB.INFINITY)
m.setObjective(u_i.sum()-muhat_iy @ phi_iy_k @ lambda_k,
                sense=grb.GRB.MINIMIZE)
m.addConstr(sp.kron( sp.eye(I),np.ones((Y,1)) ) @ u_i 
                >= phi_iy0+phi_iy_k @ lambda_k)
m.optimize()
# Displaying results:
print('lambda=', lambda_k.X, 'val=',m.objval)
