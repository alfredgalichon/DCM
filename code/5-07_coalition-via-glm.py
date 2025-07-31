# Hedonic equilibrium pricing via generalized linear models
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp
from sklearn import linear_model
np.random.seed(7)
X,Y,Z = 5,4,10
n_x, m_y = np.ones(X)/X, np.ones(Y)/Y
C_x_z = np.random.uniform(size = (X,Z))
U_y_z = np.random.uniform(size = (Y,Z))
kr=sp.kron
# Computing:
dependent = np.concatenate(\
 [(n_x[:,None] * np.exp(C_x_z)).flatten(), \
 (m_y[:,None] * np.exp(-U_y_z)).flatten()])
weights = np.concatenate( [np.exp(-C_x_z).flatten() ,
                           np.exp(U_y_z).flatten()] )

R00 = kr(np.ones((X,1)),sp.eye(Z))
R01 = - kr(sp.eye(X), np.ones((Z,1) ))
R10 = - kr(np.ones((Y,1)),sp.eye(Z))
R12 = - kr(sp.eye(Y), np.ones((Z,1) )) 

regressor = sp.bmat([[ R00, R01 , None],
                      [ R10 ,None , R12]])
hedonic_as_glm = linear_model.PoissonRegressor( \
 fit_intercept=False,alpha=0)
hedonic_as_glm.fit( regressor,dependent,sample_weight=weights)
p_z  = hedonic_as_glm.coef_[:Z]
# displaying results
print('p_z=',p_z )