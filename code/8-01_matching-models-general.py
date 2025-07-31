# Probit matching
#
# Importing libraries and generating data:
import numpy as np
from statistics import NormalDist
phi = np.vectorize(NormalDist(mu=0, sigma=1).cdf)
phi_inv = np.vectorize(NormalDist().inv_cdf)
def ghk(L,z_j,x_i_j): # ghk simulator
    I,J = x_i_j.shape
    xhat_i_j = np.zeros((I,J))
    xhat_i_j[:,0] = x_i_j[:,0] * phi(z_j[0] / L[0,0])
    GHK_i = phi(z_j[0] / L[0,0]) * np.ones(I)
    for j in range(J-1):
        phietc_i = phi((z_j[j+1]-(L[j+1,:(j+1)][None,:]* phi_inv(xhat_i_j[:,:(j+1)])).sum(axis= 1) ) /L[j+1,j+1])
        xhat_i_j[:,j+1]=x_i_j[:,j+1] * phietc_i
        GHK_i *= phietc_i
    return GHK_i.mean()
#
def piprobit_y(U_y,sigma_eta,x_i_j):
  Y=U_y.shape[0]
  I = x_i_j.shape[0]
  pi_y = np.zeros(Y)
  for y in range(Y):
      M = np.array( [[(1. if i==j else 0.) - (1. if j==y else 0.)  for j in \
       range(Y)] for i in range(Y) if i != y])
      L=  np.linalg.cholesky(M @ sigma_eta @ M.T)
      z_j  = - M @ U_y
      pi_y[y] = ghk(L,z_j,x_i_j)
  return (pi_y)
#
X,Y = 5,3
np.random.seed(7)
phi_x_y = np.random.uniform(size=(X,Y))
n_x = np.ones(X)/X # no unassigned individuals, total masses conicide
m_y = np.ones(Y)/Y
I=J= 1000
maxiter= 1000
tol = 1e-3
eps = 0.1
unif1_i_j = np.random.uniform(size=(I,Y-1))
unif2_i_j = np.random.uniform(size=(J,X-1))
# Computing:
def muprobit (U_x_y, n_x,Sigma ,unif_i_j):
    mu_x_y = np.zeros(U_x_y.shape)
    for x in range(U_x_y.shape[0]):
        mu_x_y[x,:] = n_x[x]* piprobit_y(U_x_y[x,:],Sigma,unif_i_j)
    return mu_x_y
#
nablaG = lambda U_x_y:  muprobit(U_x_y,n_x,np.eye(Y),unif1_i_j)
nablaH = lambda V_x_y:  muprobit(V_x_y.T,m_y, np.eye(X), unif2_i_j).T
#
w_x_y = np.zeros((X,Y))
for step  in range(maxiter):
    delta_x_y = nablaG(w_x_y) - nablaH(phi_x_y - w_x_y)
    if np.abs(delta_x_y).max()<tol:
        break
    w_x_y -= eps * delta_x_y
# Displaying results
print('Converged in ',step,' steps')
print('w_x_y=',w_x_y)