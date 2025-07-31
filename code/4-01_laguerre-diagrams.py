# Laguerre diagrams
#
# Importing libraries and generating data
!pip install -U sdot==2024.12.7.1
from sdot import PowerDiagram, optimal_transport_plan, SdotPlan,SumOfDiracs
import numpy as np
#
np.random.seed( 357 )
#
Y,M = 4,2
np.random.seed(800)
xi_y_m = np.random.uniform(0,2,M*Y).reshape((-1,M))
pwd = PowerDiagram(xi_y_m / 2 )
pwd.boundaries = [[1,0,1],[0,1,1],[-1,0,0],[0,-1,0]]
pwd.plot()
#
# direct problem: given U_y, compute pi_y
def direct(U_y , verbose = True, plot=False):
  pwd.weights =  U_y + ( xi_y_m * xi_y_m).sum(axis=1)
  pi_y = pwd.cell_integrals()
  e_y = pwd.cell_barycenters().T
  if verbose:
    print('U_y =', np.around(U_y,2), '\npi_y=' , np.around(pi_y, 2)) 
    print('e_y =\n' , np.around(e_y,2))
  if plot:
    pwd.plot()
  return pi_y,e_y
#
# inverse problem: given pi_y, compute U_y
def inverse(pi_y , si0 = None, err=1e-8):
  tp = SdotPlan( source_measure = SumOfDiracs( xi_y_m / 2,  masses = pi_y  ) )
  tp.adjust_potentials()
  U_y = tp.power_diagram.weights - (xi_y_m * xi_y_m).sum(axis=1) 
  return U_y - U_y[0]
#
# displaying results
direct(np.array([2,2,0,0.7]), plot=True)
U_y = inverse(np.array([1/4]*4))
direct(U_y,verbose=False, plot=True)
