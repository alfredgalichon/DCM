# Constrained logit model
#
# Importing libraries and generating data:
import scipy.optimize as opt, numpy as np
U_y = np.array([0.2, -.3, 0.5])
pibar_y = np.array([0.3, 0.5, 0.6])

# Computing
u = -np.log(opt.brentq(lambda themu : (themu+ np.minimum(themu *np.exp(U_y),pibar_y).sum() - 1) ,0,1))

# displaying results
print('u=',u)