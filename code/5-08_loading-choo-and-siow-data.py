import numpy as np, pandas as pd
# retrieves marriage dataset from:
# Choo and Siow (2006). Who marries whom and why.
# Journal of Political Economy.
def load_marriage_data(nbCateg = 25):
  thepath ='https://raw.githubusercontent.com/math-econ-code/'
  thepath += 'mec_datasets/main/marriage-ChooSiow/'
  n_singles=pd.read_csv(thepath+'n_singles.txt',
    sep='\t',header=None)
  marr = pd.read_csv(thepath+'marr.txt',sep='\t',header=None)
  muhat_x0 = np.array(n_singles[0].iloc[0:nbCateg])
  muhat_0y = np.array(n_singles[1].iloc[0:nbCateg])
  muhat_xy = np.array(marr.iloc[0:nbCateg:,0:nbCateg])
  return muhat_xy, muhat_x0, muhat_xy

muhat_xy,muhat_x0, muhat_0y= load_marriage_data()