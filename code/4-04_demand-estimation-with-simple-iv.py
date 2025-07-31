# Demand estimation with simple IV regression
#
# importing libraries and generating data
import numpy as np
#
def generate(T= 5000, seed = 77):
  np.random.seed(seed)
  p_t = np.random.uniform(size = T) 
  kappa_t = 2 * np.random.normal(size = T)
  zeta_t = np.random.normal(size = T)
  p_t = kappa_t + zeta_t
  pihat_t = np.exp( 2. - 0.5 * p_t +  kappa_t ) / (1 + np.exp( 2. - 0.5 * p_t +  kappa_t ))
  return pihat_t, p_t, zeta_t
#
T = 500
pihat_t, p_t, zeta_t = generate(T)
#
# defining instrumental variable regression
def iv(y_i,x_i_k,z_i_k):
  return np.linalg.solve( z_i_k.T @ x_i_k , y_i @ z_i_k )
#
# market share inversion
U_t = np.log(pihat_t / (1 - pihat_t))
regressors_t_k =  np.stack((np.ones(T), p_t), axis = 1 )
instruments_t_k = np.stack((np.ones(T), zeta_t),axis = 1 )
#
lambdaOLS_k = iv(U_t, regressors_t_k, regressors_t_k )
lambdaIV_k = iv(U_t, regressors_t_k, instruments_t_k )
#
# displaying results
print('OLS estimate without IV:\n', 'lambda_OLS=',lambdaOLS_k)
print('IV estimate:\n', 'lambda_IV =', lambdaIV_k)