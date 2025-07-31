# Classes of constraint choice problems
#
import numpy as np, scipy.optimize as opt, gurobipy as grb
# define parent class:
class ChoiceProblem:
  def __init__(self,n_x,alpha_x_y):
    self.n_x = n_x
    self.alpha_x_y = alpha_x_y
    self.X,self.Y = alpha_x_y.shape
#
# child class 1: no heterogeneity (individual problem)
#
class IndividualChoiceProblem(ChoiceProblem):  
  def choice( self, constraint = None):
    if constraint is None:
      mubar_x_y = self.n_x[:,None] * np.ones((self.X,self.Y))
    else:
      mubar_x_y = constraint
    alpha_x_y = np.block([[self.alpha_x_y,np.zeros( (self.X,1 )) ]])
    mu_x_y = np.zeros((self.X,self.Y+1))
    u_x = np.zeros(self.X)
    for x in range(self.X):
        nxres = self.n_x[x]
        for yind in np.argsort(alpha_x_y[x,:])[::-1]:
            if yind == self.Y:
                mu_x_y[x,yind] = nxres
                break
            if mubar_x_y[x,yind] > 0:
                mu_x_y[x,yind] = min(nxres , mubar_x_y[x,yind])
                nxres -= mu_x_y[x,yind]
            if nxres == 0:
                break
        u_x[x]= alpha_x_y[x,yind]
    return mu_x_y[:,:-1] 
#
# child class 2: logit heterogeneity
#
class LogitChoiceProblem(ChoiceProblem):  
  def choice( self, constraint = None):
    if constraint is None:
      mubar_x_y = self.n_x[:,None] * np.ones((self.X,self.Y))
    else:
      mubar_x_y = constraint
    mu_x_y = np.zeros((self.X,self.Y))
    for x in range(self.X):
      mu0 = opt.brentq(f=  lambda mu: mu+ np.minimum(mu*np.exp(self.alpha_x_y[x,:]), mubar_x_y[x,:]).sum() -self.n_x[x] ,a=0,b=self.n_x[x])
      mu_x_y[x,:] =  np.minimum(mu0*np.exp(self.alpha_x_y[x,:]), mubar_x_y[x,:])
    return mu_x_y 
#
# child class 3: simulated heterogeneity
#
class SimulatedChoiceProblem(ChoiceProblem):  
  def __init__(self,n_x,alpha_x_y,varepsilon_i_y):
    super().__init__(n_x,alpha_x_y)
    self.varepsilon_i_0 = varepsilon_i_y[:,-1]
    self.varepsilon_i_y = varepsilon_i_y[:,0:-1]
    self.I,_ = varepsilon_i_y.shape


  def choice( self, constraint = None):
    if constraint is None:
      mubar_x_y = self.n_x[:,None] * np.ones((self.X,self.Y))
    else:
      mubar_x_y = constraint
    m = grb.Model()
    m.setParam('OutputFlag', 0)
    tau_x_y = m.addMVar((self.X,self.Y))
    u_i_x = m.addMVar((self.I,self.X), lb = - grb.GRB.INFINITY)
    m.setObjective( (tau_x_y* mubar_x_y).sum() + (self.n_x[None,:]* u_i_x).sum()/self.I, sense=grb.GRB.MINIMIZE )
    mu_i_x_y = m.addConstr(u_i_x[:,:,None] >= self.alpha_x_y[None,:,:] - tau_x_y[None,:,:] + self.varepsilon_i_y[:,None,:] )
    m.addConstr(u_i_x >= self.varepsilon_i_0[:,None])
    m.optimize()
    mu_x_y = mu_i_x_y.pi.sum(axis=0)
    return(mu_x_y )