# Multinomial logistic regression w L1 regularization
#
# Importing libraries and generating data :
import numpy as np , scipy . sparse as sp
np . random . seed (7)
I ,Y , K = 100 ,5 ,3
muhat_iy = np.random.multinomial(1,[1/ Y]* Y, size =I).flatten()
phi_iy_k = np.random.normal(size = (I*Y, K))
# Define soft thresholding: 
def soft_thresholding(x, eta):
    return np.sign(x) * np.maximum(np.abs(x) - eta, 0)
# Computing :
gamma = 5.0
alpha = 1 / np.linalg.norm(phi_iy_k @ phi_iy_k.T)
lambda_k = np.zeros(K)
tol = 1e-5
iter = 1000
for i in range(iter):
  expPhi_i_y = np.exp(phi_iy_k @ lambda_k).reshape( (I,Y))
  mu_iy = (expPhi_i_y / expPhi_i_y.sum(axis = 1)[:,None]).flatten()
  grad_k =  ((muhat_iy - mu_iy) @ phi_iy_k).flatten()
  lambdatilde_k = lambda_k + alpha*grad_k
  new_lambda_k = soft_thresholding(lambdatilde_k , alpha*gamma )
  if np.linalg.norm( lambda_k -new_lambda_k) < alpha*tol:
    iter = i
    break
  lambda_k = new_lambda_k
# Displaying results:
print(iter, lambda_k)