# Multinomial logistic regression with small noise
# with the log-sum-exp trick
#
# Importing libraries and generating data :
import numpy as np
from scipy.optimize import minimize

np.random.seed(7)
I, Y, K = 500, 4, 3
pihat_iy = np.random.multinomial(1,[1/ Y ]* Y, size =I).flatten()
phi_iy_k = np.random.normal(size = (I*Y, K))
phi_iy0 = np.random.normal( size = I*Y )
sigma = 0.001
tol,iter = 1e-3, 1000000
# Computing:

def f(lambda_k):
  philambda_iy = phi_iy_k @ lambda_k
  argexpPhi_i_y = (phi_iy0 + philambda_iy).reshape( (I,Y)) / sigma 
  maxargexpPhi_i = argexpPhi_i_y.max(axis = 1)
  expPhi_i_y = np.exp( argexpPhi_i_y - maxargexpPhi_i[:,None] )
  pi_iy = (expPhi_i_y / expPhi_i_y.sum(axis = 1)[:,None]).flatten()
  return sigma * (np.log(expPhi_i_y.sum(axis =1)).sum() + maxargexpPhi_i.sum()) - pihat_iy.dot(philambda_iy )

def grad_f(lambda_k):
  philambda_iy = phi_iy_k @ lambda_k
  argexpPhi_i_y = (phi_iy0 + philambda_iy).reshape( (I,Y)) / sigma 
  maxargexpPhi_i = argexpPhi_i_y.max(axis = 1)
  expPhi_i_y = np.exp( argexpPhi_i_y - maxargexpPhi_i[:,None] )
  pi_iy = (expPhi_i_y / expPhi_i_y.sum(axis = 1)[:,None]).flatten()
  return (pi_iy - pihat_iy ) @ phi_iy_k

res = minimize(f,x0 = np.zeros(K), jac = grad_f, method='BFGS', options={'gtol': tol})
# displaying results:
print(sigma, res.x) 
