# -*- coding: utf-8 -*-
# -------------------------- CQF Module 3 Exam Code -------------------------- #
# -------------------------------- Apirl 2018 -------------------------------- #
# -------------------------------- QUAN YUAN  -------------------------------- #
# Description: Visualization

# import python package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
sns.set_style('white')

# import module coded by myself
import simulation
import fd
# ---------------------------------------------------------------------------- #
# ------------------ Part I: Underlying stock price Simulation --------------- #
# ---------------------------------------------------------------------------- #
# set parameters
# inital price
s0 = 100
# expiry
expiry = 1
# number of simulation
num_sim = 50
# time step
time_step = 252
# volatility
vol = 0.2
# risk free rate
rf = 0.05

# simulation
sim = simulation.Simulation(s0 = s0, expiry = expiry, num_sim = num_sim, \
                                time_step = time_step, vol = vol, rf = rf)
result = sim.get_result()
result = pd.DataFrame(result).T

# plot
plt.figure(figsize = (12, 8))
plt.xlabel('Time step', fontsize = 16)
plt.ylabel('Underlying stock price', fontsize = 16)
plt.plot(result.index, result, label = 'simulation')
plt.show()

# ---------------------------------------------------------------------------- #
# ------- Part II: Difference between theoretical and simulated price -------- #
# ---------------------------------------------------------------------------- #
# Notice: please run main.py first for generating xlsx file
theor_call = 10.450583572185565
theor_put = 5.573526022256971
sim_num_chg = pd.read_excel('sim_num_chg.xlsx')
sim_num_chg['call_error'] = sim_num_chg['call_sim_chg'] - theor_call
sim_num_chg['put_error'] = sim_num_chg['put_sim_chg'] - theor_put
sim_num_chg.to_csv('sim_num_chg_error.csv')

# Notice that xtick is not plot well, so we don't set xtick
plt.figure(figsize = (12, 8))
plt.plot(sim_num_chg['call_error'], label = 'put error')
plt.plot(sim_num_chg['put_error'], label = 'call error')
plt.xlabel('The number of simulation', fontsize = 16)
plt.ylabel('Call and put option error', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()

step_num_chg = pd.read_excel('time_step_chg.xlsx')
step_num_chg['call_error'] = step_num_chg['call_step_chg'] - theor_call
step_num_chg['put_error'] = step_num_chg['put_step_chg'] - theor_put
step_num_chg.to_csv('time_step_chg_error.csv')

# Notice that xtick is not plot well, so we don't set xtick
plt.figure(figsize = (12, 8))
mpl.rc('font',family='Times New Roman')
plt.plot(step_num_chg['call_error'], label = 'call error')
plt.plot(step_num_chg['put_error'], label = 'put error')
plt.xlabel('The number of time step', fontsize = 16)
plt.ylabel('Call and put option error', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()

# ---------------------------------------------------------------------------- #
# ---------------- Part III: Asian Option Continuous Sampling ---------------- #
# ---------------------------------------------------------------------------- #
# change number of simulation
num_sim = 1
# simulation
sim = simulation.Simulation(s0 = s0, expiry = expiry, num_sim = num_sim, \
                                time_step = time_step, vol = vol, rf = rf)
result = sim.get_result()
result = pd.DataFrame(result).T
result.columns = ['simulation']

con_arithmetric = []
con_geometric = []

for i in range(len(result)):
    con_arithmetric.append(result.iloc[:i, :].mean())
    con_geometric.append(result.iloc[:i,:].apply(np.log).mean().apply(np.exp))
result['con_arithmetric'] = con_arithmetric
result['con_geometric'] = con_geometric

plt.figure(figsize = (12, 8))
plt.plot(result.index, result['simulation'], label = 'simulation')
plt.plot(result.index, result['con_arithmetric'], label = 'con_arithmetric')
plt.plot(result.index, result['con_geometric'], label = 'con_geometric')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Underlying stock price', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()

# ---------------------------------------------------------------------------- #
# ------------------ Part IV: Asian Option Discrete Sampling ----------------- #
# ---------------------------------------------------------------------------- #
sample_window = 20
dis_arithmetric = []
dis_geometric = []

# get index so we know which data should be updated into average
sample_index = [i for i in range(time_step) if i%sample_window == 0]

A_arith = result.iloc[0]['simulation']
A_geo = result.iloc[0]['simulation']
count = 0
for i in range(len(result)):
    if i%sample_window == 0 and i != 0:
        # if true, it is time to update
        count += 1
        A_arith = (1.0*result.iloc[i]['simulation']/count) + ((count-1)*A_arith*1.0/count)
        A_geo = np.exp(1.0*np.log(result.iloc[i]['simulation'])/count + \
                       (count - 1)*np.log(A_geo)*1.0/count)
    dis_arithmetric.append(A_arith)
    dis_geometric.append(A_geo)
result['dis_arithmetric'] = dis_arithmetric
result['dis_geometric'] = dis_geometric

plt.figure(figsize = (12, 8))
plt.plot(result.index, result['simulation'], label = 'simulation')
plt.plot(result.index, result['dis_arithmetric'], label = 'dis_arithmetric')
plt.plot(result.index, result['dis_geometric'], label = 'dis_geometric')
plt.xlabel('Time', fontsize = 16)
plt.ylabel('Underlying stock price', fontsize = 16)
plt.legend(fontsize = 16)
plt.show()

# ---------------------------------------------------------------------------- #
# -------- Part V: Finite Difference method for European and US option ------- #
# ---------------------------------------------------------------------------- #
# 5.1 European option
FDeuro = fd.FD(vol = 0.2, rf = 0.05, expiry = 1, strike = 100, \
                      option_type = 'call', etype = 'N', nas = 100)
euro_option_call = FDeuro.price(visual = True)

euro_asset, euro_time = np.meshgrid(sorted(list(set(euro_option_call['asset']))), \
                          sorted(list(set(euro_option_call['time']))))
euro_value = np.reshape(euro_option_call['value'], euro_asset.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(euro_asset, euro_time, euro_value, cmap=cm.YlGnBu_r)
ax.set_xlabel(r'$S$')
ax.set_ylabel(r'$T-t$')
ax.set_zlabel(r'$C(S,t)$')
plt.show() 

# 5.2 Aerican option
FDus = fd.FD(vol = 0.2, rf = 0.05, expiry = 1, strike = 100, \
                      option_type = 'call', etype = 'Y', nas = 100)
us_option_call = FDus.price(visual = True)

us_asset, us_time = np.meshgrid(sorted(list(set(us_option_call['asset']))), \
                          sorted(list(set(us_option_call['time']))))
us_value = np.reshape(us_option_call['value'], us_asset.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(us_asset, us_time, us_value, cmap=cm.YlGnBu_r)
ax.set_xlabel(r'$S$')
ax.set_ylabel(r'$T-t$')
ax.set_zlabel(r'$C(S,t)$')
plt.show()

# 5.3 Option value table

# European call
FDeuro_call = fd.FD(vol = 0.2, rf = 0.05, expiry = 1, strike = 100, \
                      option_type = 'call', etype = 'N', nas = 100)
euro_option_call = FDeuro_call.price(visual = False)
print "# ------------------- FD Method for European call ------------------- #"
print euro_option_call[euro_option_call['asset'] == 100]['value']

# European put
FDeuro_put = fd.FD(vol = 0.2, rf = 0.05, expiry = 1, strike = 100, \
                      option_type = 'put', etype = 'N', nas = 100)
euro_option_put = FDeuro_put.price(visual = False)
print "# ------------------- FD Method for European put -------------------- #"
print euro_option_put[euro_option_put['asset'] == 100]['value']

# American call
FDus_call = fd.FD(vol = 0.2, rf = 0.05, expiry = 1, strike = 100, \
                      option_type = 'call', etype = 'Y', nas = 100)
us_option_call = FDus_call.price(visual = False)
print "# ------------------- FD Method for American call ------------------- #"
print us_option_call[us_option_call['asset'] == 100]['value']

# American put
FDus_put = fd.FD(vol = 0.2, rf = 0.05, expiry = 1, strike = 100, \
                      option_type = 'put', etype = 'Y', nas = 100)
us_option_put = FDus_put.price(visual = False)
print "# ------------------- FD Method for American put -------------------- #"
print us_option_put[us_option_put['asset'] == 100]['value']
