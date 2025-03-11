import numpy as np
(T,X,Y,K) = 5,3,4,2
np.random.seed(7)
phi_t_x_y_k = np.random.uniform(size=(T,X,Y,K))
muhat_t_x_y = np.random.uniform(size=(T,X,Y))
