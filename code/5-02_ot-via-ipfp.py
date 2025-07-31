# Optimal transport via IPFP
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp
np.random.seed(7)
X,Y = 5,4
n_x, m_y = np.ones(X)/X, np.ones(Y)/Y
phi_xy = np.random.uniform(size = X*Y)
tol,maxit = 1e-5,1000
# Computing:
K_x_y = np.exp(phi_xy).reshape((X,Y))
B_y = np.ones(Y)
for i in range(maxit):
  A_x = n_x / (K_x_y @ B_y)
  Bnew_y = m_y / (A_x @ K_x_y)
  if np.abs(Bnew_y-B_y).max() < tol:
    break
  B_y = Bnew_y
u_x,v_y = - np.log(A_x),- np.log(B_y)
# displaying results
print('Value=',u_x @ n_x +v_y @ m_y , 'Steps=',i)
