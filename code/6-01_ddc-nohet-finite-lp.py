 # Dynamic discrete choice via LP
#
# Importing libraries and generating data:
!pip install gurobipy
import numpy as np, gurobipy as grb
T,X,Y = 40,3,2
beta = 0.9
phi_t_x_y = beta**np.arange(T)[:,None,None] \
            * np.array( [[[25,-10], [12,-10 ],[-5,-10]]])
n_x = np.array([10,20,5])
P_xprime_x_y = np.array([[[1/3, 1],[0, 1],[0, 1]], \
 [[2/3, 0],[1/3,0],[0, 0]],[[0,0],[2/3, 0],[1, 0]]])
c = 9 # size of the workshop
# Computing:
m = grb.Model()
mu_t_x_y = m.addMVar((T,X,Y))
m.setObjective( (mu_t_x_y* phi_t_x_y).sum(), 
               sense = grb.GRB.MAXIMIZE)
u_0_x = m.addConstr( n_x == mu_t_x_y[0,:,:].sum(axis = 1))
u_t_x = m.addConstr( (mu_t_x_y[:-1,None,:,:] * P_xprime_x_y[ \
 None,:,:,:]).sum(axis=(2,3))== mu_t_x_y[1:,:,:].sum(axis=2))
lambda_t = m.addConstr( mu_t_x_y[:,:,1].sum(axis = 1) <= c )
m.optimize()
# displaying results
u_0_x.pi
\end{lstlisting}
\subsection{No Heterogeneity, Finite Horizon via Backward Induction}
\label{app:ddc-nohet-finite-bwd-fwd-induc}
\begin{lstlisting}
# Dynamic discrete choice via backward-forward induction
#
# Importing libraries and generating data:
import numpy as np, gurobipy as grb
T,X,Y = 40,3,2
beta = 0.9
phi_x_y = np.array( [[25,-10], [12,-10 ],[-5,-10]])
phi_t_x_y = np.array([beta**t for t in range(T)])[:,None,None] *  phi_x_y[None,:,:]
n_x = np.array([10,20,5])
P_xprime_x_y = np.array([[[1/3, 1],[0, 1],[0, 1]],[[2/3, 0],[1/3,0],
                        [0, 0]],[[0,0],[2/3, 0],[1, 0]]])
# Computing -- backward:
u_t_x = np.zeros((T,X))
u_t_x[T-1,:] = phi_t_x_y[T-1,:,:].max(axis = 1)
for t in range(T-2,-1,-1):
  u_t_x[t,:] = ( phi_t_x_y[t,:,:] + (P_xprime_x_y * u_t_x[t+1,:][:,None,None]).sum(axis = 0) ).max(axis = 1 )
# Computing -- forward
mu_t_x_y = np.zeros((T,X,Y))
then_x = n_x
for t in range(T-1):
  y_x = ( phi_t_x_y[t,:,:] + 
  (P_xprime_x_y * u_t_x[t+1,:][:,None,None]).sum(axis = 0) ).argmax(axis = 1 )
  mu_t_x_y[t,list(range(X)),list(y_x) ]=then_x
  then_x = P_xprime_x_y.reshape((X,X*Y)) @ mu_t_x_y[t,:,:].flatten()
y_x = ( phi_t_x_y[T-1,:,:] ).argmax(axis = 1 )
mu_t_x_y[T-1,list(range(X)),list(y_x) ]= then_x
# Displaying results
print('u_t_x=',u_t_x[0,:])
print('objective=' ,(mu_t_x_y * phi_t_x_y ).sum() )