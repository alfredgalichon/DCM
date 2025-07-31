import numpy as np
(X,Y,Omega) = 5,5,3
np.random.seed(7)
M_y_omega=np.random.uniform(size=(Y,Omega))
n_x = np.ones(X)
P_y = np.ones(Y)
tau_x_y = np.random.uniform(size=(X,Y))
np.fill_diagonal(tau_x_y, 1)
zeta_x_omega = np.random.uniform(size=(X,Omega))