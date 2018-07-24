# -*- coding: utf-8 -*-
"""
# CQF Module 5 Exam Problem 1 (2)
@author: QUAN YUAn
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
import matplotlib as mpl

def p12_merton(x):
    # initialization
    v0 = x[0]
    
    d1 = 1.0*(np.log(v0*1.0/debt) + (rf + 1.0*sigma_v**2/2)*T)/(sigma_v*np.sqrt(T))
    d2 = d1 - sigma_v*np.sqrt(T)
    equ1 = v0*norm.cdf(d1) - debt*np.exp(-rf*T)*norm.cdf(d2)
    # loss function
    return [equ0 - equ1]

if __name__ == '__main__':
    # initialization
    
    equ0 = 3 # million
    debt = 5 # million
    T = 1 # year
    rf = 0.02 # 2%
    k = 5 # million
    merton_pd = []
    black_pd = []
    v0_list = []
    sigma_v_list = []
    sigma_v_list = np.arange(0.1, 0.9, 0.01)
    
    for sigma_v in sigma_v_list:
        # merton model
        v0 = fsolve(p12_merton, x0 = [3.15])[0]
        v0_list.append(v0)
        print sigma_v, v0
        
        # merton model
        merton_d1 = 1.0*(np.log(v0*1.0/debt) + \
                         (rf + 1.0*sigma_v**2/2)*T)/(sigma_v*np.sqrt(T))
        merton_d2 = merton_d1 - sigma_v*np.sqrt(T)
        merton_pd.append(norm.cdf(-merton_d2))
        
        # black and cox model
        black_h1 = (np.log(k*1.0/(np.exp(rf*T)*v0)) + T*sigma_v**2/2)/(sigma_v*np.sqrt(T))
        black_h2 = black_h1 - sigma_v*np.sqrt(T)
        black_pd.append(norm.cdf(black_h1) + \
                        np.exp(2*(rf - (sigma_v**2/2))*np.log(k*1.0/v0)/sigma_v**2)*norm.cdf(black_h2))
        
    # visulization
    plt.figure(figsize = (12, 8))
    mpl.rc('font',family='Times New Roman')
    plt.plot(sigma_v_list, merton_pd, label = 'Merton Model', linewidth = 4)
    plt.plot(sigma_v_list, black_pd, label = 'Black-Cox Model', linewidth = 4)
    plt.xlabel('volatility', fontsize = 16)
    plt.ylabel('probability of default', fontsize = 16)
    plt.title('PD Merton vs Black-Cox', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()
    
    # save dataframe
    black_df = pd.DataFrame({'v0': v0_list, 'sigma': sigma_v_list, 'Theo_PD': black_pd})
    black_df.to_excel('black_theo.xlsx')
    