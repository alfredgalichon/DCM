 # Dynamic discrete choice via LP
#
# Importing libraries and generating data:
#!pip install gurobipy
import numpy as np, gurobipy as grb
T,X,Y = 40,3,2
beta = 0.9
phi_t_x_y = beta**np.arange(T)[:,None,None] \
            * np.array( [[[25,-10], [12,-10 ],[-5,-10]]])
n_x = np.array([10,20,5])
P_xprime_x_y = np.array([[[1/3, 1],[0, 1],[0, 1]], \
 [[2/3, 0],[1/3,0],[0, 0]],[[0,0],[2/3, 0],[1, 0]]])
c = 9 # size of the workshop
# Computing:
m = grb.Model()
mu_t_x_y = m.addMVar((T,X,Y))
m.setObjective( (mu_t_x_y* phi_t_x_y).sum(), 
               sense = grb.GRB.MAXIMIZE)
u_0_x = m.addConstr( n_x == mu_t_x_y[0,:,:].sum(axis = 1))
u_t_x = m.addConstr( (mu_t_x_y[:-1,None,:,:] * P_xprime_x_y[ \
 None,:,:,:]).sum(axis=(2,3))== mu_t_x_y[1:,:,:].sum(axis=2))
lambda_t = m.addConstr( mu_t_x_y[:,:,1].sum(axis = 1) <= c )
m.optimize()
# displaying results
u_0_x.pi