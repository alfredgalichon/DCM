# Demand inversion via linear programming
#
# Importing libraries and generating data:
import numpy as np, gurobipy as grb
np.random.seed(7)
I,Y = 100,3
mu_y = np.array([27,19,25,29]) / I
epsilon_iy = np.random.normal(size=I*(Y+1) )
# Computing:
m = grb.Model()
u_i = m.addMVar(I, lb = - grb.GRB.INFINITY)
U_y = m.addMVar(Y+1, lb = - grb.GRB.INFINITY)
m.setObjective( u_i.sum()/I - U_y @ mu_y, 
                sense = grb.GRB.MINIMIZE)
m.addConstr(  np.kron(np.eye(I),np.ones((Y+1,1))) @ u_i 
            - np.kron(np.ones((I,1)),np.eye(Y+1)) @ U_y 
            >= epsilon_iy)
m.addConstr( U_y[0]==0 )
m.optimize()
# Displaying results:
print( 'U_y=' , U_y.x)

