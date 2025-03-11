import numpy as np, pandas as pd
#
def load_travel_data():
  thepath = 'https://raw.githubusercontent.com/math-econ-code/mec_datasets/main/demand_travelmode/'
  the_data = pd.read_csv(thepath+'travelmodedata.csv')
  pihat_iy = np.where(the_data['choice'] =='yes' , 1, 0)
  Y = the_data['mode'].nunique()
  I = the_data.shape[0] // Y
  covariates = the_data[['travel', 'income', 'gcost']].values
  Phi_iy_k = np.column_stack([ covariates[:,0] , - (covariates[:,0] 
  * covariates[:,1] ), - covariates[:,2] ])
  _,K = Phi_iy_k.shape
  Phibar_k = Phi_iy_k.mean(axis = 0)
  Phistdev_k = Phi_iy_k.std(axis = 0, ddof = 1)
  Phi_iy_k = ((Phi_iy_k - Phibar_k[None,:]) / Phistdev_k[None,:])
  return I,Y,K,pihat_iy,Phi_iy_k

I,Y,K,pihat_iy,Phi_iy_k = load_travel_data()