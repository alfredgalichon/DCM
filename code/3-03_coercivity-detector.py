# Coercivity detector
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp, gurobipy as grb
np.random.seed(7)
I,Y,K = 100,5,3
muhat_i_y = np.random.multinomial(1, [1/Y]*Y, size=I)
phi_i_y_k = np.random.normal( size = (I,Y,K))
# Computing:
m=grb.Model()
t = m.addMVar(1)
mubar_i_y = m.addMVar( (I,Y))
m.setObjective(t, sense=grb.GRB.MAXIMIZE)
m.addConstr( mubar_i_y >= t )
m.addConstr( mubar_i_y.sum(axis=1) == muhat_i_y.sum(axis=1) )
m.addConstr( (mubar_i_y[:,:,None]*phi_i_y_k).sum(axis=(0,1) ) == (muhat_i_y[:,:,None]*phi_i_y_k).sum(axis=(0,1) ) )

m.optimize()
# Displaying results:
print('V=', t.X)