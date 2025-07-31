import numpy as np, pandas as pd
# retrieves trade dataset from:
# Yotov et al. (2016). An Advanced Guide to Trade Policy 
# Analysis: The Structural Gravity Model. WTO.
def get_trade_data():
  thepath = 'https://raw.githubusercontent.com/math-econ-code/'
  thepath += 'mec_datasets/main/gravity_wtodata/'
  tradedata = pd.read_csv(thepath + \
   '1_TraditionalGravity_from_WTO_book.csv') 
  tradedata = tradedata[['exporter', 'importer','year', \
   'trade', 'DIST','ln_DIST', 'CNTG', 'LANG', 'CLNY']]
  tradedata.sort_values(['year','exporter','importer'], 
                        inplace = True)
  tradedata.reset_index(inplace = True, drop = True)
  nbt = len(tradedata['year'].unique())
  nbi = len(tradedata['importer'].unique())
  nbk = 4
  years = tradedata['year'].unique() 
  distances = np.array(['ln_DIST', 'CNTG', 'LANG', 'CLNY']) 
  D_x_y_t_k = np.zeros((nbi,nbi,nbt,nbk)) 
  tradevol_x_y_t = np.zeros((nbi,nbi,nbt)) 
  muhat_x_y_t = np.zeros((nbi,nbi,nbt)) 
  for t, year in enumerate(years):
      tradevol_x_y_t[:, :, t] = np.array(tradedata.loc[ \
       tradedata['year'] == year, 'trade']).reshape((nbi, nbi)) 
      np.fill_diagonal(tradevol_x_y_t[:, :, t], 0)  
      for k, distance in enumerate(distances):
          D_x_y_t_k[:, :, t, k] = np.array(tradedata.loc[ \
           tradedata['year'] == year, distance]).reshape( \
           (nbi, nbi))  
  muhat_x_y_t=tradevol_x_y_t/(tradevol_x_y_t.sum()/len(years))
  return D_x_y_t_k, muhat_x_y_t

D_x_y_t_k,muhat_x_y_t = get_trade_data()