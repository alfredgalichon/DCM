import numpy as np
from sklearn.linear_model import LogisticRegression
(I,Y,L) = (1000,5,7)
np.random.seed(1)
Psi_i_l = np.random.uniform(size = (I,L))
pihat_i_y = np.random.multinomial(1, np.ones(Y) / Y,size = I)
yhat_i = pihat_i_y.argmax (axis = 1)
model = LogisticRegression( max_iter=100000, fit_intercept=False, penalty = None, tol=1e-8 )
model.fit(Psi_i_l, yhat_i)
theta_y_l = model.coef_
print((theta_y_l - theta_y_l[0,:][None,:]).T)
