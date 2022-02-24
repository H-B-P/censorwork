import pandas as pd
import numpy as np

import math

c_mean=2
k_mean=1

c_sig=2
k_sig=1

df = pd.read_csv('normal.csv')

lr=0.07/len(df)
nrounds=10000

def erf(x):
    # save the sign of x
    sign = np.sign(x)
    x = abs(x)
    
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)


def meandiff(x, u, s):
 return (x-u)/(s*s)

def sigdiff(x, u, s):
 return ((x-u)**2 - s**2)/(s**3)

def meandiff_censored(x, u, s):
 PDF = (1/s)*math.sqrt(2/math.pi)*np.exp(-(x-u)*(x-u)/(2*s*s))
 CDF = (1 + erf((x-u)/(s*math.sqrt(2))))
 return PDF/CDF

def sigdiff_censored(x, u, s):
 return meandiff_censored(x, u, s)*((x-u)/s)

for i in range(nrounds):
 sigs = c_sig + df['x']*k_sig
 means = c_mean + df['x']*k_mean
 
 c_mean += sum(meandiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]]))*lr
 k_mean += sum(meandiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]])*df['x'][~df["censored"]])*lr
 c_sig += sum(sigdiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]]))*lr
 k_sig += sum(sigdiff(df['y'][~df["censored"]], means[~df["censored"]], sigs[~df["censored"]])*df['x'][~df["censored"]])*lr
 #print(i, c_mean, k_mean, c_sig, k_sig)
 c_mean -= sum(meandiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]]))*lr
 k_mean -= sum(meandiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]])*df['x'][df["censored"]])*lr
 c_sig -= sum(sigdiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]]))*lr
 k_sig -= sum(sigdiff_censored(df['y'][df["censored"]], means[df["censored"]], sigs[df["censored"]])*df['x'][df["censored"]])*lr
 print(i, c_mean, k_mean, c_sig, k_sig)
 
