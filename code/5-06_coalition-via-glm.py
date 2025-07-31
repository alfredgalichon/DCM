# Coalition formation via GLM
#
# Importing libraries and generating data:
import numpy as np, scipy.sparse as sp
from sklearn import linear_model as lm
np.random.seed(7)
x_log_x = np.vectorize(lambda x: x * np.log(x) if x > 0 else 0)
Z,K = 4,3
B = Z*(Z+1) // 2
# in this example, same-sex marriage: coalitions of size <= 2
count_b_z = np.array([ list(np.eye(Z)[z1] + np.eye(Z)[z2])  
          for z2 in range(Z) for z1 in range(Z) if z1<= z2 ])
muhat_b = np.random.uniform(size=B)
phi_b_k = np.random.uniform(size = (B,K) )
# Computing :
s_b = count_b_z.sum(axis = 1)
R_b_p = np.block([phi_b_k , count_b_z ]) /  s_b[:,None]
l_b =  (x_log_x(count_b_z) ).sum(axis = 1) / s_b
glm = lm.PoissonRegressor(fit_intercept=False,alpha=0)
glm.fit( R_b_p, muhat_b * np.exp(l_b) , 
        sample_weight = s_b * np.exp(-l_b) )
# displaying results
print('lambda_k=',glm.coef_[:K])