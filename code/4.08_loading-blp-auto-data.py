# Loading BLP's automobile data
import pyblp, numpy as np,  pandas as pd
#
product_data = pd.read_csv(pyblp.data.BLP_PRODUCTS_LOCATION)
#
agent_data = pd.read_csv(pyblp.data.BLP_AGENTS_LOCATION)
#
