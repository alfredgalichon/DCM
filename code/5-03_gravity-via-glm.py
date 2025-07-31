# Gravity via GLMs
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp
from sklearn import linear_model as lm
np.random.seed(7)
X,Y,K = 5,5,3
muhat_x_y = np.random.uniform(size = (X,Y) )
np.fill_diagonal(muhat_x_y,0)
phi_xy_k = np.random.uniform(size = (X*Y,3) )
# Computing:
regr = sp.hstack([phi_xy_k , 
                    -sp.kron(sp.eye(X), np.ones((Y, 1))), 
                    -sp.kron(np.ones((X, 1)),sp.eye(Y))])
w_xy = (np.ones((X,Y)) - np.eye(X)).flatten()
clf = lm.PoissonRegressor(fit_intercept=False, alpha=0)
clf.fit(regr, muhat_x_y.flatten(), sample_weight=w_xy)
# displaying results
print('lambda_k=',clf.coef_[:K])