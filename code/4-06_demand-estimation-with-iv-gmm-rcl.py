# Demand estimation with IV-GMM, random coefficient logit case
#
# importing libraries and generating data
import numpy as np
from scipy.optimize import fsolve, minimize

#
def generate(T= 5000, I=10000, seed = 77):
  np.random.seed(seed)
  p_t = np.random.uniform(size = T)
  kappa_t = 2 * np.random.normal(size = T)
  zeta_t = np.random.normal(size = T)
  zetap_t = np.random.normal(size = T)
  zetapp_t = np.random.normal(size = T)
  eta_i = - np.exp( - (1 + np.random.normal(size = I ) ) )
  p_t = kappa_t + 0.3* zeta_t + 0.5 * zetap_t + 0.2 * zetapp_t
  #u_i_t =  2. - 0.5 * p_t[None,:] +  0.1 * eta_i[:,None]*p_t[None,:] + kappa_t[None,:] 
  #pihat_t = (np.exp( u_i_t) / (1 + np.exp( u_i_t ))).mean(axis=0)
  pihat_t = np.exp( 2. - 0.5 * p_t + kappa_t  ) / (1 + np.exp( 2. - 0.5 * p_t  + kappa_t ))
  return pihat_t, p_t, zeta_t, zetap_t,zetapp_t, eta_i
#
T,I = 500,1000
pihat_t, p_t, zeta_t, zetap_t,zetapp_t,eta_i = generate(T,I)
#
# inversing the random coefficient logit model
def valUhat_t(tau):
  U_t = np.zeros(T)
  for t in range(T):
    thef = lambda U: (1/(1+np.exp(U+tau*eta_i*p_t[t]))).mean() - 1+ pihat_t[t]
    U_t[t] = fsolve(thef, np.log(pihat_t[t] / (1-pihat_t[t])))[0]
  return U_t

def derUhat_t(tau,Uhat_t = None):
  if Uhat_t is None:
    Uhat_t = valUhat_t(tau)
  epsilonp_i_t = eta_i[:,None]*p_t[None,:] 
  denom_i_t = 1 + np.exp( Uhat_t + tau*epsilonp_i_t )
  return  (epsilonp_i_t / denom_i_t).mean(axis=0) /(1-pihat_t)
#
# defining IV-GMM regression
def nl_iv_gmm(W_l_l = None):
  if W_l_l is None:
    W_l_l = np.linalg.inv( ins_t_l.T @ ins_t_l)
  Pi_t_t = ins_t_l @ W_l_l @ ins_t_l.T
  Gamma_t_t = Pi_t_t - Pi_t_t @ reg_t_k @ np.linalg.inv(reg_t_k.T @ Pi_t_t @ reg_t_k) @ reg_t_k.T @ Pi_t_t
  #
  def F(tau):
    Uhat_t = valUhat_t(tau)
    return Uhat_t @ Gamma_t_t @ Uhat_t
  #
  def derF(tau):
    Uhat_t = valUhat_t(tau)
    dUdtau_t = derUhat_t(tau, Uhat_t)
    return 2* Uhat_t @ Gamma_t_t @ dUdtau_t
  #
  tauhat = minimize(F, 1, jac = derF).x[0]
  Uhat_t = valUhat_t(tauhat)
  lambdahat_k = np.linalg.solve(reg_t_k.T @ Pi_t_t @ reg_t_k, Uhat_t @ Pi_t_t @ reg_t_k)
  kappahat_t = Uhat_t - reg_t_k @ lambdahat_k
  return (tauhat, lambdahat_k, kappahat_t)
#
reg_t_k = np.stack((np.ones(T), p_t), axis = 1)
ins_t_l = np.stack ((np.ones(T), zeta_t,zetap_t,zetapp_t), axis = 1)
#
tau1, lam1_k, kap1_t = nl_iv_gmm()
kappaT_t_l = kap1_t[:,None]*ins_t_l
W_l_l = T*  np.linalg.inv(kappaT_t_l.T @ kappaT_t_l)
tau2, lam2_k, kap2_t = nl_iv_gmm(W_l_l )
#
# displaying results
print('IV-GMM 1st stage:\n', 'tau_1S', tau1, 'lambda_1S=',lam1_k)
print('IV-GMM 2nd stage:\n', 'tau_2S', tau2, 'lambda_2S=',lam2_k)
