# -*- coding: utf-8 -*-
# -------------------------- CQF Module 3 Exam Code -------------------------- #
# -------------------------------- Apirl 2018 -------------------------------- #
# -------------------------------- QUAN YUAN  -------------------------------- #
# Description: explore European option

# import python package
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import matplotlib as mpl

# import module code by myself
import bsm
import monte_carlo
import simulation

# function for calculating the BSM theoretical price(both call and put)
def bsm_euro(s, strike, expiry, vol, rf, show = False):
    euro_bsm = bsm.Bsm_european(s = s, strike = strike, expiry = expiry, vol = vol, rf = rf)
    theoretical_call = euro_bsm.call_value()
    theoretical_put = euro_bsm.put_value()
    if show == True:
        print 'The theoretical Black-Scholes value for call option is ', round(theoretical_call, 3)
        print 'The theoretical Black-Scholes value for put option is ', round(theoretical_put, 3)
        print 'put call parity test: ', theoretical_put + s == theoretical_call + s*np.exp(-rf*expiry)
    return [theoretical_call, theoretical_put]


# function for calculating the simulated price(both call and put)
def sim_euro(s0, strike, expiry, num_sim, time_step, vol, rf, show = False):
    # simulation uderlying stock price
    sim = simulation.Simulation(s0 = s0, expiry = expiry, num_sim = num_sim, \
                                time_step = time_step, vol = vol, rf = rf)
    result = sim.get_result()
    # change type to dataframe
    result = pd.DataFrame(result).T
    # get call and put option price
    euro_call_sim = monte_carlo.Sim_european(sim_result = result, strike = strike, \
                                         rf = rf, expiry = expiry, option_type = 'call')
    euro_put_sim = monte_carlo.Sim_european(sim_result = result, strike = strike, \
                                         rf = rf, expiry = expiry, option_type = 'put')
    if show == True:
        result.iloc[:, :50].plot(title = 'Monte Carlo simulation for stock price', \
                             legend = False, \
                             figsize = (12, 8))
        print "The simulation price for European call option is ",euro_call_sim.get_value()
        print "The simulation price for European put option is ",euro_put_sim.get_value()
    return [euro_call_sim.get_value(), euro_put_sim.get_value()]

if __name__ == '__main__':
    
    # ---------------------------------------------------------------------------- #
    # --------- Part I: Comparsion between theoretical and simulated price ------- #
    # ---------------------------------------------------------------------------- #
    
    print "# ----------------- BSM theoretical price for option ------------------ #"
    bsm_euro(s = 100, strike = 100, expiry = 1, vol = 0.2, rf = 0.05, show = True)
    
    print "# ------------ Monte Carlo simulation for European Option ------------- #"
    # 40 different number of simulations
    sens_num_sim = np.hstack([np.arange(5, 200, 5), np.arange(200, 5000, 200)])
    # 39 different number of time steps
    sens_time_step = np.hstack([np.arange(1, 30, 1), np.arange(30, 2000, 50)])
    # for different simulation times
    print "Change the number of simulations"
    # list for storing call and put option price
    call_sim_chg = []
    put_sim_chg = []
    # TIME
    start = time.time()
    for i in sens_num_sim:
        [call_value, put_value] = sim_euro(s0 = 100, strike = 100, expiry = 1, num_sim = i, \
                                        time_step = 252, vol = 0.2, rf = 0.05, show = False)
        call_sim_chg.append(call_value)
        put_sim_chg.append(put_value)
    end = time.time()
    print "Finished and the time consume is ",round(end - start, 2),'s'
    
    # for different number of time step
    print "Change the number of time steps"
    call_step_chg = []
    put_step_chg = []
    start = time.time()
    for i in sens_time_step:
        [call_value, put_value] = sim_euro(s0 = 100, strike = 100, expiry = 1, num_sim = 500, \
                                        time_step = i, vol = 0.2, rf = 0.05, show = False)
        call_step_chg.append(call_value)
        put_step_chg.append(put_value)
    end = time.time()
    print "Finished and time consume is ",round(end - start, 2),'s'
    
    # store the result in 
    pd.DataFrame({'sim_num': sens_num_sim, 'call_sim_chg': call_sim_chg, \
                  'put_sim_chg': put_sim_chg}).to_excel('sim_num_chg.xlsx')
    pd.DataFrame({'step_num': sens_time_step, 'call_step_chg': call_step_chg, \
                  'put_step_chg': put_step_chg}).to_excel('time_step_chg.xlsx')
    
    # ---------------------------------------------------------------------------- #
    # --------- Part II: Relationship between option price and parameters -------- #
    # ---------------------------------------------------------------------------- #
    
    # 2.1 Strike
    strike_list = np.arange(50, 150, 1)
    call_value_bsm = []
    put_value_bsm = []
    call_value_sim = []
    put_value_sim = []
    for each_strike in strike_list:
        call_value, put_value = bsm_euro(s = 100, strike = each_strike, expiry = 1, \
                                   vol = 0.2, rf = 0.05, show = False)
        call_value_bsm.append(call_value)
        put_value_bsm.append(put_value)
        
        call_value, put_value = sim_euro(s0 = 100, strike = each_strike, expiry = 1, \
                                    num_sim = 1000, time_step = 1, vol = 0.2, \
                                    rf = 0.05, show = False)
        call_value_sim.append(call_value)
        put_value_sim.append(put_value)
        
    plt.figure(figsize = (8, 10))
    mpl.rc('font',family='Times New Roman')
    plt.plot(strike_list, call_value_bsm, label = 'call option bsm value')
    plt.plot(strike_list, put_value_bsm, label = 'put option bsm value')
    plt.plot(strike_list, call_value_sim, label = 'call option simulated value')
    plt.plot(strike_list, put_value_sim, label = 'put option simulated value')
    plt.xlabel('Strike Price', fontsize = 16)
    plt.ylabel('Call/Put option value', fontsize = 16)
    plt.title('European option value with different strike', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()
    
    # 2.2 volatility
    vol_list = np.arange(0.01, 0.9, 0.005)
    call_value_bsm = []
    put_value_bsm = []
    call_value_sim = []
    put_value_sim = []
    for each_vol in vol_list:
        call_value, put_value = bsm_euro(s = 100, strike = 100, expiry = 1, \
                                   vol = each_vol, rf = 0.05, show = False)
        call_value_bsm.append(call_value)
        put_value_bsm.append(put_value)
        
        call_value, put_value = sim_euro(s0 = 100, strike = 100, expiry = 1, \
                                    num_sim = 1000, time_step = 1, vol = each_vol, \
                                    rf = 0.05, show = False)
        call_value_sim.append(call_value)
        put_value_sim.append(put_value)
        
    plt.figure(figsize = (8, 10))
    plt.plot(vol_list, call_value_bsm, label = 'call option bsm value')
    plt.plot(vol_list, put_value_bsm, label = 'put option bsm value')
    plt.plot(vol_list, call_value_sim, label = 'call option simulated value')
    plt.plot(vol_list, put_value_sim, label = 'put option simulated value')
    plt.xlabel('Volatility', fontsize = 16)
    plt.ylabel('Call/Put option value', fontsize = 16)
    plt.title('European option value with different volatility', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()
    
    # 2.3 risk-free rate
    rf_list = np.arange(0.01, 0.9, 0.005)
    call_value_bsm = []
    put_value_bsm = []
    call_value_sim = []
    put_value_sim = []
    for each_rf in rf_list:
        call_value, put_value = bsm_euro(s = 100, strike = 100, expiry = 1, \
                                   vol = 0.2, rf = each_rf, show = False)
        call_value_bsm.append(call_value)
        put_value_bsm.append(put_value)
        
        call_value, put_value = sim_euro(s0 = 100, strike = 100, expiry = 1, \
                                    num_sim = 1000, time_step = 1, vol = 0.2, \
                                    rf = each_rf, show = False)
        call_value_sim.append(call_value)
        put_value_sim.append(put_value)
        
    plt.figure(figsize = (8, 10))
    plt.plot(rf_list, call_value_bsm, label = 'call option bsm value')
    plt.plot(rf_list, put_value_bsm, label = 'put option bsm value')
    plt.plot(rf_list, call_value_sim, label = 'call option simulated value')
    plt.plot(rf_list, put_value_sim, label = 'put option simulated value')
    plt.xlabel('Interest rate', fontsize = 16)
    plt.ylabel('Call/Put option value', fontsize = 16)
    plt.title('European option value with different interest rate', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()
    
    # 2.4 Expiry
    expiry_list = np.arange(0.1, 5, 0.01)
    call_value_bsm = []
    put_value_bsm = []
    call_value_sim = []
    put_value_sim = []
    for each_expiry in expiry_list:
        call_value, put_value = bsm_euro(s = 100, strike = 100, expiry = each_expiry, \
                                   vol = 0.2, rf = 0.05, show = False)
        call_value_bsm.append(call_value)
        put_value_bsm.append(put_value)
        
        call_value, put_value = sim_euro(s0 = 100, strike = 100, expiry = each_expiry, \
                                    num_sim = 1000, time_step = 1, vol = 0.2, \
                                    rf = 0.05, show = False)
        call_value_sim.append(call_value)
        put_value_sim.append(put_value)
        
    plt.figure(figsize = (8, 10))
    plt.plot(expiry_list, call_value_bsm, label = 'call option bsm value')
    plt.plot(expiry_list, put_value_bsm, label = 'put option bsm value')
    plt.plot(expiry_list, call_value_sim, label = 'call option simulated value')
    plt.plot(expiry_list, put_value_sim, label = 'put option simulated value')
    plt.xlabel('Expiry', fontsize = 16)
    plt.ylabel('Call/Put option value', fontsize = 16)
    plt.title('European option value with different expiry', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()