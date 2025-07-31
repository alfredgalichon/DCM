# Cupids via GLM
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp
from sklearn import linear_model as lm
kr = sp.kron
np.random.seed(7)
X,Y,K = 5,5,3
A = X*Y+X+Y
muhat_a = np.random.uniform(size = A )
phi_xy_k = np.random.uniform(size = (X*Y,K) )
# Computing :
s_a = np.array([2.0]*(X*Y)+[1.0]*(X+Y))
R00 = phi_xy_k/2
R01 = -kr(sp.eye(X),np.ones((Y,1)))/2
R02 = -kr(np.ones((X,1)),sp.eye(Y) )/2
R11 = - sp.eye(X)
R22 = - sp.eye(Y)
R_a_p = sp.bmat([[R00,R01,R02],
                 [None,-sp.eye(X),None],
                 [None,None,-sp.eye(Y)]])
glm = lm.PoissonRegressor(fit_intercept=False,alpha=0)
glm.fit( R_a_p, muhat_a , sample_weight = s_a )
lambda_k  = glm.coef_[:K]
# displaying results
print('lambda_k=',lambda_k)