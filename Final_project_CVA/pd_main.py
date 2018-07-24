# -*- coding: utf-8 -*-
"""
CQF Final Project CDS Bootstrapping
@author: QUAN YUAN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import pd_bootstrapping
# load data
CS = pd.read_csv('CreditSpread.csv')
del CS['Unnamed: 0']
DF = pd.read_csv('DF.csv')
DF = DF[DF['tensor'] != 0.08]
del DF['Unnamed: 0']

# parameters setting
n = 10
RR = 0.4
dt = 0.5
T = DF.tensor.values
DF = DF.DF.values

# Interpolation for credit spread
cs_inter = pd_bootstrapping.Interpolation(data = {'x': CS.tensor.values, \
                                 'y': CS.cs.values/10000})
cs_result = cs_inter.get(x_list = T[1:], fun = lambda x: x, rev_fun = lambda x: x)
cs_result = cs_result

# cds bootsrapping
# survival prob
survival_prob = pd_bootstrapping.prob_survival(DF, RR, dt, cs_result, n)
# prob of default
prob_default = -np.diff(survival_prob)
prob_default = np.insert(prob_default, 0, 0)
# cumlative prob of default
prob_default_cum = 1 - survival_prob
# implied lambda
lam = [0]
for i, j in zip(T[1:], survival_prob[1:]):
    lam.append(-np.log(j)*1.0/i)
    
# save result
all_result = pd.DataFrame({'tensor': T, 'cs_result': [0] + cs_result, \
                       'DF': DF, 'lam': lam, 'PD': prob_default,\
                       'PD_cum': prob_default_cum, 'P': survival_prob})
all_result.to_csv('PD_result.csv')
all_result.index = list(all_result.tensor)

# visualization
# term structure PD and lambda
ax1 = all_result[['PD', 'lam']].iloc[1:].plot(kind = 'bar', width=0.8, \
          title = 'Term Structure of Lambda and PD')
plt.xlabel('Tensor')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
plt.show()

# cumulative distribution
ax2 = all_result[['P', 'PD_cum']].plot(marker='o', \
          title = 'Cumulative distributions', ylim=(0, 1.1))
plt.xlabel('Tensor')
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
plt.show()


