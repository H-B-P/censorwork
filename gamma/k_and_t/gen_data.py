import numpy as np
import pandas as pd

import random

c_k = 4
m_k = 0.5

c_t = 6
m_t = 0.5

censor_mean = 30
censor_sig = 1

possible_xes = [0,1,2,3,4,5,6,7,10]

nrows = 10000

x = np.random.choice(possible_xes, nrows)


k = c_k + m_k*x
t = c_t + m_t*x

true_y = np.random.gamma(k,t,nrows)
censor_y = np.random.normal(censor_mean, censor_sig, nrows)

dfDict = {'x':x, 'true_y':true_y, 'censor_y': censor_y}

df = pd.DataFrame(dfDict)

df['y'] = df[['true_y','censor_y']].max(axis=1)

df['censored'] = df['censor_y']>df['true_y']

df.to_csv('gamma.csv')
