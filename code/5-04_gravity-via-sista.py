# Gravity via SISTA
# Importing libraries and generating data:
import numpy as np
np.random.seed(7)
X,Y,K = 5,5,3
muhat_x_y = np.random.uniform(size = (X,Y) )
np.fill_diagonal(muhat_x_y,0)
phi_xy_k = np.random.uniform(size = (X*Y,3) )
def soft_thresholding(x, eta):
    return np.sign(x) * np.maximum(np.abs(x) - eta, 0)
# Computing :
n_x,m_y = muhat_x_y.sum(axis =1), muhat_x_y.sum(axis =0)
gamma = 0.1 # set to zero for no l1 penalty
alpha = 1 / np.linalg.norm(phi_xy_k @ phi_xy_k.T)
lambda_k = np.zeros(K)
tol = 1e-5
iter = 1000
B_y = np.ones(Y)
for i in range(iter):
  K_x_y = np.exp(phi_xy_k @ lambda_k).reshape( (X,Y))
  np.fill_diagonal(K_x_y,0)
  A_x = n_x / (K_x_y @ B_y)
  B_y = m_y / (A_x @ K_x_y)
  mu_x_y =  (A_x[:,None] * B_y[None,:] * K_x_y)
  grad_k =  (muhat_x_y - mu_x_y).flatten() @ phi_xy_k
  lambdatilde_k = lambda_k + alpha*grad_k
  new_lambda_k = soft_thresholding(lambdatilde_k , alpha*gamma )
  if np.linalg.norm( lambda_k -new_lambda_k) < alpha*tol:
    iter = i
    break
  lambda_k = new_lambda_k
# displaying results
print('lambda_k=',lambda_k,'Steps=',i)