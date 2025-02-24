# Market share simulation
#
# Importing libraries and generating data:
import numpy as np
np.random.seed(7)
I,Y = 100,3
U_y = np.array([0.0, 0.9,0.7,0.3]) 
epsilon_i_y = np.random.normal(size=(I,Y+1))
# Computing:
u_i_y = U_y[None,:]+epsilon_i_y
u_i = u_i_y.max(axis = 1)
G = u_i.sum() / I
mu_i_y = 1* ( u_i[:,None] == u_i_y )
mu_y = mu_i_y.sum(axis = 0 ) / I
# Displaying results:
print('G(U)=',G, ';', 'mu_y=', mu_y)
