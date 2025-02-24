# A max-factor model
#
# Importing libraries and generating the parameters:
import numpy as np
#
np.random.seed(7)
O = 10000
X = 10000000
thelambda = .8
epsilon_o = np.zeros(O)
epsilonprime_o = np.zeros(O)

denom = (np.arange(1,X+1) ** (-thelambda) ).sum() 
U_x = np.log( (np.arange(1,X+1) ** (-thelambda) )  / denom )
Uprime_x = np.log(  (np.arange(X,0,-1) ** (-thelambda) ) / denom )

rho = 1-thelambda*thelambda

for o in range(O):
    eta_x = - np.log( np.random.uniform(size=X)) 
    epsilon_o[o] = (U_x + eta_x).max()
    epsilonprime_o[o] = (Uprime_x + eta_x).max()
    if (1+o) % 100 == 0:
        print(np.corrcoef(epsilon_o[:o], epsilonprime_o[:o])[0,1] , rho)