 # Data generation for the estimation of dynamic matching models
import numpy as np
from scipy import optimize, sparse as sp
from scipy.sparse import kron
np.random.seed(7)
X,Y,K = 15,12,3
B = X*Y+X+Y
beta = 0.9
rel_tol = 1e-5
epsilon = 0.05
max_it = 10000
phi_b_k = np.random.normal(size=(B,K))
muhat_b = np.random.uniform(size=B)
P_xprime_b = np.random.uniform(size= (X,B))
P_xprime_b = P_xprime_b / P_xprime_b.sum(axis=0)[None,:]
Q_yprime_b = np.random.uniform(size= (Y,B))
Q_yprime_b = Q_yprime_b / Q_yprime_b.sum(axis=0)[None,:]