# Nested logit inversion
#
# Importing libraries and generating data:
import numpy as np
#
Y = 10
X = 3
pi_y = np.array([0.05,0.12,0.08,0.21,0.17,0.02,0.14,0.10,0.07, 0.04])
nest_y = np.array([0,2,1,1,0,2,1,2,2,0])
lambda_x = np.array([0.3, 0.5, 0.8])
# computing the fixed effect matrix
in_x_y = 1* (np.arange(X)[:,None] == nest_y[None,:])
# computing the market share of nests
pix_y = (in_x_y @ pi_y) @ in_x_y
lambdax_y = lambda_x @ in_x_y
# inverse market share map:
U_y = lambdax_y * np.log(pi_y) + (1 - lambdax_y) * np.log( pix_y  )
# entropy of choice:
Gstar = lambdax_y * pi_y * np.log(pi_y) + (1 - lambdax_y) * pix_y * np.log( pix_y  )
#
# displaying results
print('G*=',Gstar)
print('U_y=',U_y)
