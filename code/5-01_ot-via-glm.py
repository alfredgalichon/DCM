# Optimal transport via generalized linear models
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp 
from sklearn import linear_model as lm
np.random.seed(7)
X,Y = 5,4
n_x, m_y = np.ones(X)/X, np.ones(Y)/Y 
phi_xy = np.random.uniform(size = X*Y)
# Computing:
w_xy = np.exp( phi_xy )
muhat_xy =  np.kron(n_x,m_y) / w_xy
ot_as_glm = lm.PoissonRegressor(fit_intercept=False,alpha=0)
regr = - sp.hstack([ sp.kron(sp.eye(X), np.ones((Y,1))),
                    sp.kron(np.ones((X,1)),sp.eye(Y))])
ot_as_glm.fit( regr, muhat_xy , sample_weight = w_xy )
u_x,v_y  = ot_as_glm.coef_[:X], ot_as_glm.coef_[X:]
# displaying results
print('value=',u_x @ n_x +v_y @ m_y )
