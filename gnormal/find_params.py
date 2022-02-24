import pandas as pd
import numpy as np

import math

import scipy
from scipy.special import erf

c_u=100
m_u=0

c_p=0.2
m_p=0

df = pd.read_csv('gnormal.csv')

u_lr=1/len(df)
p_lr=0.0002/len(df)

nrounds=10000



def u_diff(y, u, p):
 return y*(y-u)/((p**2)*(u**3)) - 1/u

def p_diff(y, u, p):
 return (y-u)**2/((p**3)*(u**2)) - 1/p

def PDF(y, u, p):
 return np.exp(-0.5*((y-u)/(p*u))**2) / (p*u*math.sqrt(2*math.pi))

def CDF(y, u, p):
 return 0.5*(1 - erf((y-u)/(p*u*math.sqrt(2))))

def u_diff_censored(y, u, p):
 #print( (y/u)*PDF(y,u,p)/CDF(y,u,p) )
 return (y/u)*PDF(y,u,p)/CDF(y,u,p)

def p_diff_censored(y, u, p):
 #print( ((y-u)/p),PDF(y,u,p),CDF(y,u,p) )
 return ((y-u)/p)*PDF(y,u,p)/CDF(y,u,p)

cenDf = df[df['censored']].reset_index()
unCenDf = df[~df['censored']].reset_index()
print(len(cenDf), len(unCenDf))

for i in range(nrounds):
 df['u'] = c_u + df['x']*m_u
 df['p'] = c_p + df['x']*m_p
 
 cenDf = df[df['censored']].reset_index()
 unCenDf = df[~df['censored']].reset_index()
 
 c_u += sum(u_diff(unCenDf['y'],unCenDf['u'],unCenDf['p']))*u_lr
 m_u += sum(u_diff(unCenDf['y'],unCenDf['u'],unCenDf['p'])*unCenDf['x'])*u_lr
 #print(sum(p_diff(unCenDf['y'],unCenDf['u'],unCenDf['p']))*p_lr)
 c_p += sum(p_diff(unCenDf['y'],unCenDf['u'],unCenDf['p']))*p_lr
 m_p += sum(p_diff(unCenDf['y'],unCenDf['u'],unCenDf['p'])*unCenDf['x'])*p_lr
 
 c_u += sum(u_diff_censored(cenDf['y'],cenDf['u'],cenDf['p']))*u_lr
 m_u += sum(u_diff_censored(cenDf['y'],cenDf['u'],cenDf['p'])*cenDf['x'])*u_lr
 #print(sum(p_diff_censored(cenDf['y'],cenDf['u'],cenDf['p']))*p_lr)
 c_p += sum(p_diff_censored(cenDf['y'],cenDf['u'],cenDf['p']))*p_lr
 m_p += sum(p_diff_censored(cenDf['y'],cenDf['u'],cenDf['p'])*cenDf['x'])*p_lr
 
 #c_u += sum(u_diff(df['true_y'],df['u'],df['p']))*lr
 #m_u += sum(u_diff(df['true_y'],df['u'],df['p'])*df['x'])*lr
 #c_p += sum(p_diff(df['true_y'],df['u'],df['p']))*lr
 #m_p += sum(p_diff(df['true_y'],df['u'],df['p'])*df['x'])*lr
 
 print(i, [c_u, m_u, c_p, m_p])
 
