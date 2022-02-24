import pandas as pd
import numpy as np

import math

import scipy
from scipy.special import psi

c_k=3
m_k=1

c_t=3
m_t=1

df = pd.read_csv('gamma.csv')

lr=0.1/len(df)
nrounds=10000

def k_diff(ykt):
 y, k, t = ykt
 return -psi(k) - np.log(t) + np.log(y)

def t_diff(ykt):
 y, k, t = ykt
 return (y - k*t)/(t**2)

def k_diff(y, k, t):
 return -psi(k) - np.log(t) + np.log(y)

def t_diff(y, k, t):
 return (y - k*t)/(t**2)


#def meandiff_censored(x, u, s):
# PDF = (1/s)*math.sqrt(2/math.pi)*np.exp(-(x-u)*(x-u)/(2*s*s))
# CDF = (1 + erf((x-u)/(s*math.sqrt(2))))
# return PDF/CDF
#
#def sigdiff_censored(x, u, s):
# return meandiff_censored(x, u, s)*((x-u)/s)

for i in range(nrounds):
 df['k'] = c_k + df['x']*m_k
 df['t'] = c_t + df['x']*m_t
 
 #c_k += sum(df[['true_y','k','t']].apply(k_diff, axis=1))*lr
 #m_k += sum(df[['true_y','k','t']].apply(k_diff, axis=1)*df['x'])*lr
 #c_t += sum(df[['true_y','k','t']].apply(t_diff, axis=1))*lr
 #m_t += sum(df[['true_y','k','t']].apply(t_diff, axis=1)*df['x'])*lr
 
 c_k += sum(k_diff(df['true_y'],df['k'],df['t']))*lr
 m_k += sum(k_diff(df['true_y'],df['k'],df['t'])*df['x'])*lr
 c_t += sum(t_diff(df['true_y'],df['k'],df['t']))*lr
 m_t += sum(t_diff(df['true_y'],df['k'],df['t'])*df['x'])*lr
 
# c_mean += sum(meandiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]]))*lr
# k_mean += sum(meandiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]])*df['x'][~df["censored"]])*lr
# c_sig += sum(sigdiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]]))*lr
# k_sig += sum(sigdiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]])*df['x'][~df["censored"]])*lr
 #print(i, c_mean, k_mean, c_sig, k_sig)
# c_mean -= sum(meandiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]]))*lr
# k_mean -= sum(meandiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]])*df['x'][df["censored"]])*lr
# c_sig -= sum(sigdiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]]))*lr
# k_sig -= sum(sigdiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]])*df['x'][df["censored"]])*lr
 print(i, c_k, m_k, c_t, m_t)
 
