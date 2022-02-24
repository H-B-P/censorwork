import pandas as pd
import numpy as np

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Gauss_hess(pred,act):
 return 2

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Poisson_hess(pred,act):
 return act/(pred*pred)

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)

def Gamma_hess(pred,act):
 return (2*act-pred)/(pred*pred*pred)

def gnormal_u_diff(y, u, p):
 return (y*(y-u)/((p**2)*(u**3)) - 1/u

def gnormal_p_diff(y, u, p):
 return (y-u)**2/((p**3)*(u**2)) - 1/p
