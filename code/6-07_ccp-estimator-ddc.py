# CCP estimator for dynamic discrete choice
#
# Importing libraries and generating data:
import numpy as np
from scipy import optimize, sparse as sp
np.random.seed(7)
X,Y,K = 10,2,3
beta = 0.9
rel_tol = 1e-6
epsilon = 0.02
max_it = 1e6
phi_x_y_k = np.random.normal(size=(X,Y,K))/10
muhat_xy = np.random.uniform(size=(X*Y))
P_xprime_xy = np.random.uniform(size= (X,X*Y))
P_xprime_xy = P_xprime_xy / P_xprime_xy.sum(axis=0)[None,:]
# Computing:
Psi = beta* P_xprime_xy.T - np.kron( np.eye(X),np.ones((Y,1)))
deltaone = np.kron(np.eye(X), np.array([[1.0]+[0.0]*(Y-1)])) 
A = np.eye(X*Y) + Psi @ np.linalg.inv( deltaone @ Psi ) @ deltaone 
logpihat_xy = np.log(muhat_xy.reshape((X,Y)) / muhat_xy.reshape((X,Y)).sum(axis = 1)[:,None]).flatten()
phihat_xy = A @ logpihat_xy
# display results:
print(phihat_xy)