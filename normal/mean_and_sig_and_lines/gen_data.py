import numpy as np
import pandas as pd

import random

c_mean = 4
k_mean = 0.5


c_sig = 6
k_sig = 0.5

possible_xes = [0,1,2,3,4,5,6,7,10]

nrows = 10000

x = np.random.choice(possible_xes, nrows)
mean = c_mean + k_mean*x
sig = c_sig + k_sig*x

dfDict = {'x':x, 'y':list(np.random.normal(mean,sig,nrows))}
df = pd.DataFrame(dfDict)

df.to_csv('normal.csv')
