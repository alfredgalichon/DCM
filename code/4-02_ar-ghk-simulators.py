# Geweke, Hajivassiliou and Keane simulator
#
# Importing libraries and generating data:
import numpy as np
from statistics import NormalDist
phi = np.vectorize(NormalDist(mu=0, sigma=1).cdf)
phi_inv = np.vectorize(NormalDist().inv_cdf)
np.random.seed(9)
I,B,J=100000,10,2
L = np.array([[np.random.uniform()*(i <= j) for i in range(J)] for j in range(J)]  )
z_j = np.random.normal(size=J)/100
x_b_i_j = np.random.uniform(size=(B,I,J))
# Coding the simulators:
def iar(L,z_j,x_i_j): # integrated accept-reject simulator
  cond_i_j = (phi_inv(x_i_j)[:,None,:] * L[None,:,:]).sum(axis = 2) <= z_j[ None,:]
  argphi_i = (z_j[-1] - (L[-1,:-1][None,:] * phi_inv(x_i_j[:,:-1])).sum(axis =1) ) / L[-1,-1]
  return (( cond_i_j[:,:-1]).all(axis = 1) * phi( argphi_i )).mean()
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
# Running simulations:
iarSim_b = np.array([iar(L,z_j,x_b_i_j[b,:,:]) for b in range(B)])
ghkSim_b = np.array([ghk(L,z_j,x_b_i_j[b,:,:]) for b in range(B)])
# Displaying results:
print('IAR, mean=',iarSim_b.mean(),'; std=',iarSim_b.std())
print('GHK, mean=',ghkSim_b.mean(),'; std=',ghkSim_b.std())
