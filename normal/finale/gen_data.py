import numpy as np
import pandas as pd

import random

c_mean = 4
k_mean = 0.5

c_sig = 6
k_sig = 0.5

censor_mean = 4
censor_sig = 1

possible_xes = [0,1,2,3,4,5,6,7,10]

nrows = 100000

x = np.random.choice(possible_xes, nrows)


mean = c_mean + k_mean*x
sig = c_sig + k_sig*x

true_y = np.random.normal(mean,sig,nrows)
censor_y = np.random.normal(censor_mean, censor_sig, nrows)

dfDict = {'x':x, 'true_y':np.random.normal(mean,sig,nrows), 'censor_y': np.random.normal(censor_mean, censor_sig, nrows)}

df = pd.DataFrame(dfDict)

df['y'] = df[['true_y','censor_y']].max(axis=1)

df['censored'] = df['censor_y']>df['true_y']

df.to_csv('normal.csv')
