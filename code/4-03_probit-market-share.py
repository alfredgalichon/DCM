# Probit market share 
#
# Importing libraries and generating data
import numpy as np
from statistics import NormalDist
phi = np.vectorize(NormalDist(mu=0, sigma=1).cdf)
phi_inv = np.vectorize(NormalDist().inv_cdf)
np.random.seed(9)
I,Y=100000,5
U_y = np.array([ 1.3, 0.9, -0.7, 0.2, 0.6])
x_i_j = np.random.uniform(size=(I,Y-1))
# Computing:
#
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
sigma_eta = np.eye(Y)
pi_y = piprobit_y(U_y,np.eye(Y),x_i_j)
# Displaying results:
print('pi_y=', pi_y,'\nsum(pi_y)=' ,pi_y.sum())
