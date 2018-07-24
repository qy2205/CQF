# -*- coding: utf-8 -*-
"""
CQF Final Project CVA Calculation Part
@author: QUAN YUAN
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

import cva
import hjm
# --------------------------------------------------------------------------- #
# -------------------- Data Loading and parameters setting ------------------ #
# --------------------------------------------------------------------------- #
# parameters
RR = 0.4
N = 1
dt = 0.5
T = 5
# using the most recent forward curve to get the fixed rate
# let the MtM = 0 at time 0(present)
fix = 0.01425
sim_num = 100

# Input for CVA
# discount factor table
df = pd.read_csv('DF_table.csv')
try: del df['Unnamed: 0']
except: pass
# probability of default
PD = pd.read_csv('PD_result.csv')
tensor = PD['tensor'].values
PD = PD['PD'].values

# Input for HJM
vol_df = pd.read_csv('vol.csv')
try: del vol_df['Unnamed: 0']
except: pass
drift = pd.read_csv('drift.csv')
try: del drift['Unnamed: 0']
except: pass
f0 = pd.read_csv('f0.csv')
try: del f0['Unnamed: 0']
except: pass

# --------------------------------------------------------------------------- #
# ------------- Calculate Exposure with simulated forward curve ------------- #
# --------------------------------------------------------------------------- #
# output results
# market to market results df
# exposure df
# calculate cva with dynamic method
start_time = time.time()
CVA = cva.CVA_IRS(T = T, dt = dt, N = N, RR = RR, df = df.iloc[0], \
                  PD = PD[1:], fix = fix)
sim_expo = []
sim_mtm = []
sim_cva = []
sim_pcva = []
for i in range(sim_num):
    # simulated forward curve
    HJM_MC = hjm.MC(num = 1000, step = 0.01)
    new_mc_result = HJM_MC.new_hjm(tensor = list(np.arange(0, 25.01,0.5)), \
                           f0 = f0.values.T[0], \
                           drift = np.matrix(list(drift.values)).T, \
                           vol = vol_df.values)
    # select forward curve
    mc_result = pd.DataFrame(new_mc_result, columns = list(np.arange(0,25.01,0.5)),\
                             index = np.arange(0, 10.0001, 0.01))
    mc_result = mc_result[np.arange(0,5.1,0.5)].loc[np.arange(0,5.1,0.5)]
    # calculate CVA
    MtM = CVA.mtm(fwd = mc_result)
    expo = CVA.exposure(MtM)
    cva_value = CVA.cva(expo)
    pcva_value = CVA.p_cva(expo)
    sim_expo.append(expo)
    sim_mtm.append(MtM)
    sim_cva.append(cva_value)
    sim_pcva.append(pcva_value)
end_time = time.time()
print 'CVA Calculation Running Time is: ', end_time - start_time
tenors = np.arange(0,5.1,0.5)
expo_df = pd.DataFrame(sim_expo, columns = tenors)
mtm_df = pd.DataFrame(sim_mtm, columns = tenors)
cva_df = pd.DataFrame(sim_cva, columns = ['cva'])
pcva_df = pd.DataFrame(sim_pcva, columns = tenors[1:])

# --------------------------------------------------------------------------- #
# ------------------------------ Calculate CVA ------------------------------ #
# --------------------------------------------------------------------------- #
# expected exposure
eexpo = pd.DataFrame(sim_expo).mean().values
# calculate the CVA
# mean
final_cva_value_mean = CVA.cva(eexpo)
final_cva_value_median = CVA.cva(expo_df.median().values)
final_cva_value_97 = CVA.cva(expo_df.quantile(0.975).values)
print 'final_cva_value_mean ', final_cva_value_mean
print 'final_cva_value_median ', final_cva_value_median
print 'final_cva_value_97 ', final_cva_value_97
# --------------------------------------------------------------------------- #
# ------------------------------ Visualization ------------------------------ #
# --------------------------------------------------------------------------- #
# simulated exposure
plt.plot(expo_df.T.index, eexpo, 'r', linewidth = 2, label = 'mean')
plt.plot(expo_df.T.index, expo_df.median(), 'g', linewidth = 2, label = 'median')
plt.plot(expo_df.T.index, expo_df.quantile(0.975), 'b', linewidth = 2, label = '97.5%')
plt.legend()
plt.title('Simulated IRS Exposure')
plt.xlabel('Tensors')
plt.show()

expo_df.T.plot(legend = False, alpha = 1)
plt.title('Simulated IRS Exposure')
plt.xlabel('Tensors')
plt.show()
# exposure amd mtm distribution
for i in range(len(tenors)-1):
    plt.subplot(5,2,i+1)
    plt.hist(mtm_df[tenors[i+1]], label = 'MTM')
    plt.hist(expo_df[tenors[i+1]], label = 'Exposure')
    plt.title('Year '+ str(tenors[i+1]), fontsize = 14)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
#    plt.axis('off')
    plt.tight_layout()
plt.show()
# simulated mtm
mtm_df.T.plot(legend = False)
plt.title('Simulated IRS Market-to-Market price')
plt.xlabel('Tensors')
plt.show()
# simulated CVA distribution
sns.distplot(sim_cva)
plt.title('Simulated IRS CVA Distribution')
plt.ylabel('probability')
plt.xlabel('CVA value')
plt.show()
# period CVA
sns.boxplot(pcva_df)
plt.title('Box plot for period CVA')
plt.xlabel('tenors')
plt.show()
# exposure describe
expo_df.describe().to_csv('exposure_stat.csv')







