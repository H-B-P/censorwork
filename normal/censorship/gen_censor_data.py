import numpy as np
import pandas as pd

import random

trueMu = 4
trueSigma= 7
censorMu = 1
censorSigma = 1

nrows = 20000

dfDict = {'truemean':list(np.random.normal(trueMu,trueSigma,nrows)),'lowercensor':list(np.random.normal(censorMu, censorSigma, nrows))}
df = pd.DataFrame(dfDict)

df['seenResponse'] = df.max(axis=1)

df['censored'] = df['lowercensor']>df['truemean']

df.to_csv('normal.csv')
