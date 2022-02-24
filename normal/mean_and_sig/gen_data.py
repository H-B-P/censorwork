import numpy as np
import pandas as pd

import random

trueMu = 4
trueSigma= 0.5

nrows = 10000

dfDict = {'true':list(np.random.normal(trueMu,trueSigma,nrows))}
df = pd.DataFrame(dfDict)

df.to_csv('normal.csv')
