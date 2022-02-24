import pandas as pd
import numpy as np

import math

mean=0
lr=0.001
nrounds=100

df = pd.read_csv('normal.csv')

sigma=7

def gimmediff(x, u, s):
 PDF = (1/s)*math.sqrt(2/math.pi)*np.exp(-(x-u)*(x-u)/(2*s*s))
 CDF = (1 + math.erf((x-u)/(s*math.sqrt(2))))
 return PDF/CDF

for i in range(nrounds):
 oldmean = mean
 for r in range(len(df)):
  if(df['censored'][r]):
   pass
   mean -= gimmediff(df['seenResponse'][r], oldmean, sigma)*lr
  else:
   pass
   mean -= (oldmean-df['seenResponse'][r])/(sigma*sigma)*lr
 print(i, mean)
