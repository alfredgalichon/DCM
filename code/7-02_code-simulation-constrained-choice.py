# Constrained choice simulation
#
# Importing libraries and generating data:
!pip install gurobipy
import numpy as np, gurobipy as grb
np.random.seed(7)
I,Y = 100,3
U_y = np.array([0.2, -.3, 0.5])
pibar_y = np.array([0.3, 0.2, 0.1])
epsilon_i_y = np.random.normal(size=(I,Y+1) )
# Computing:
m = grb.Model()
lambda_i_y = m.addMVar( (I,Y+1))
m.setObjective(  (lambda_i_y * ( np.array([0]+list(U_y))[None,:] + epsilon_i_y)).sum(),
               sense = grb.GRB.MAXIMIZE)
u_i = m.addConstr( lambda_i_y.sum(axis=1) == 1/I )
tau_y = m.addConstr( lambda_i_y.sum(axis=0)[1:] <= pibar_y )
m.optimize()
# Displaying results:
print( 'tau_y=' , tau_y.pi)