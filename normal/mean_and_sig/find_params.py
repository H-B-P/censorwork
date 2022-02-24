import pandas as pd
import numpy as np

import math

mean=4
sig=1

df = pd.read_csv('normal.csv')

lr=0.04/len(df)
nrounds=1000


def meandiff(x, u, s):
 return (x-u)#/(s*s)

def sigdiff(x, u, s):
 return ((x-u)**2 - s**2)/s#(s**3)


for i in range(nrounds):
 oldsig = sig
 oldmean = mean
 for r in range(len(df)):
  mean += meandiff(df['true'][r], oldmean, oldsig)*lr
  sig += sigdiff(df['true'][r], oldmean, oldsig)*lr
 print(i, mean, sig)
