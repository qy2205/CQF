# -*- coding: utf-8 -*-
"""
# CQF Module 5 Exam Problem 1 (1)
@author: QUAN YUAN
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

def p11_merton(x):
    # initialization
    v0 = x[0]
    sigma_v = x[1]
    
    d1 = 1.0*(np.log(v0*1.0/debt) + (rf + 1.0*sigma_v**2/2)*T)/(sigma_v*np.sqrt(T))
    d2 = d1 - sigma_v*np.sqrt(T)
    
    # equity and sigma of equity
    # norm.cdf default mean = 0, vol = 1
    equ1 = v0*norm.cdf(d1) - debt*np.exp(-rf*T)*norm.cdf(d2)
    sigma_equ1 = sigma_v*norm.cdf(d1)*v0*1.0/equ1
    
    # loss function
    return [equ0 - equ1, sigma_equ0 - sigma_equ1]

if __name__ == '__main__':
    equ0 = 3 # million
    sigma_equ0 = 0.5 # 50%
    debt = 5 # million
    T = 1 # year
    rf = 0.02 # 2%
    
    print 'Solution to this equation: '
    print fsolve(p11_merton, x0 = [3.79, 0.55])