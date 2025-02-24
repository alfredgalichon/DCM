# Multinomial logistic regression via GLM
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import PoissonRegressor
np.random.seed(7)
I,Y,K = 100,5,3
pihat_iy = np.random.multinomial(1,[1/Y ]*Y,size=I).flatten()
phi_iyk = np.random.normal( size = I*Y*K)
# Computing the parameter:
reg = PoissonRegressor(alpha = 0, fit_intercept=False)
R_omega_p = sp.hstack([phi_iyk.reshape(-1,K), 
                -sp.kron(sp.identity(I), np.ones((Y,1)))])
reg.fit(R_omega_p, pihat_iy  )
# Computing the inverse variance-covariance matrix:
pilambda_i_y = np.exp(R_omega_p @ reg.coef_ ).reshape((-1,Y))
phi_i_y_k = phi_iyk.reshape((-1,Y,K))
B1 = (pilambda_i_y[:,:,None,None] * phi_i_y_k[:,:,:,None] * phi_i_y_k[:,:,None,:]).sum(axis = (0,1)) / I
B2=  ( pilambda_i_y[:,:,None,None,None] * pilambda_i_y[:,None,:,None,None] * phi_i_y_k[:,:,None,:,None] * phi_i_y_k[:,None,:,None,:] ).sum(axis = (0,1,2)) / I
Bhat_k_l = B1 - B2
# Displaying results:
print('lambda=\n', reg.coef_[:K])
print('Fisher matrix=\n', np.linalg.inv(Bhat_k_l))