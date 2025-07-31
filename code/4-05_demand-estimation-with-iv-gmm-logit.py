# Demand estimation with IV-GMM, logit case
#
# importing libraries and generating data
import numpy as np
#
def generate(T= 5000, seed = 77):
  np.random.seed(seed)
  p_t = np.random.uniform(size = T) 
  kappa_t = 2 * np.random.normal(size = T)
  zeta_t = np.random.normal(size = T)
  zetaprime_t = np.random.normal(size = T)
  p_t = kappa_t + 0.3* zeta_t + 0.7 * zetaprime_t
  pihat_t = np.exp( 2. - 0.5 * p_t +  kappa_t ) / (1 + np.exp( 2. - 0.5 * p_t +  kappa_t ))
  return pihat_t, p_t, zeta_t, zetaprime_t
#
T = 500
pihat_t, p_t, zeta_t, zetaprime_t = generate(T)
#
# defining IV-GMM regression
def lambda_gmm(dep_t, reg_t_k, ins_t_l, W_l_l = None):
  if W_l_l is None:
    W_l_l = np.linalg.inv( ins_t_l.T @ ins_t_l)
  Pi_t_t = ins_t_l @ W_l_l @ ins_t_l.T
  return np.linalg.solve( reg_t_k.T @ Pi_t_t @ reg_t_k, U_t @ Pi_t_t @ reg_t_k ) 
#
# market share inversion
U_t = np.log(pihat_t / (1 - pihat_t))
regressors_t_k =  np.stack((np.ones(T), p_t), axis = 1 )
instruments_t_l = np.stack( (np.ones(T), zeta_t, zetaprime_t), axis = 1)
#
lambda1S_k = lambda_gmm(U_t,regressors_t_k, instruments_t_l )
kappahat_t = U_t - regressors_t_k @ lambda1S_k
kappaT_t_l = kappahat_t[:,None]*instruments_t_l
W_l_l = T*  np.linalg.inv(kappaT_t_l.T @ kappaT_t_l) 
lambda2S_k = lambda_gmm(U_t,regressors_t_k, instruments_t_l,W_l_l )
#
# displaying results
print('IV-GMM 1st stage:\n', 'lambda_1S=',lambda1S_k)
print('IV-GMM 2nd stage:\n', 'lambda_2S =', lambda2S_k)   
