import pandas as pd
import numpy as np

import math

c_mean=2
k_mean=2

c_sig=2
k_sig=1

df = pd.read_csv('normal.csv')

lr=0.07/len(df)
nrounds=10000


def meandiff(x, u, s):
 return (x-u)/(s*s)

def sigdiff(x, u, s):
 return ((x-u)**2 - s**2)/(s**3)

for i in range(nrounds):
 sigs = c_sig + df['x']*k_sig
 means = c_mean + df['x']*k_mean
 c_mean += sum(meandiff(df['y'], means, sigs))*lr
 k_mean += sum(meandiff(df['y'], means, sigs)*df['x'])*lr
 c_sig += sum(sigdiff(df['y'], means, sigs))*lr
 k_sig += sum(sigdiff(df['y'], means, sigs)*df['x'])*lr
  #c_mean += meandiff(df['y'][r], means[r], sigs[r])*lr
  #k_mean += meandiff(df['y'][r], means[r], sigs[r])*df['x'][r]*lr
  #c_sig += sigdiff(df['y'][r], means[r], sigs[r])*lr
  #k_sig += sigdiff(df['y'][r], means[r], sigs[r])*df['x'][r]*lr
 print(i, c_mean, k_mean, c_sig, k_sig)
