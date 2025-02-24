# Multinomial logistic regression via gradient descent
#
# Importing libraries and generating data :
import numpy as np
#
np.random.seed(7)
I, Y, K = 100, 5, 3
pihat_iy = np.random.multinomial(1,[1/ Y ]* Y, size =I).flatten()
phi_iy_k = np.random.normal(size = (I*Y, K))
# Computing:
alpha = 1 / np.linalg.norm(phi_iy_k @ phi_iy_k.T)
lambda_k = np.zeros(K)
tol,iter = 1e-3, 1000
for i in range(iter):
  expPhi_i_y = np.exp(phi_iy_k @ lambda_k).reshape( (I,Y))
  pi_iy = (expPhi_i_y / expPhi_i_y.sum(axis = 1)[:,None]).flatten()
  grad_k =  ((pihat_iy - pi_iy) @ phi_iy_k).flatten()
  if np.linalg.norm(grad_k) < tol:
    break
  lambda_k = lambda_k + alpha * grad_k
# Displaying results:
print('Iterations=', i, 'lambda_k=',lambda_k)
