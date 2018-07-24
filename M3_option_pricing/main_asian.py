# -*- coding: utf-8 -*-
# -------------------------- CQF Module 3 Exam Code -------------------------- #
# -------------------------------- Apirl 2018 -------------------------------- #
# -------------------------------- QUAN YUAN  -------------------------------- #
# Description: exploring asian option
# import python package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('white')

# import module code by myself
import monte_carlo
import simulation
 
if __name__ == '__main__':
    # ---------------------------------------------------------------------------- #
    # --------- Part I: Comparsion between theoretical and simulated price ------- #
    # ---------------------------------------------------------------------------- #
    
    # simulation uderlying stock price
    sim = simulation.Simulation(s0 = 100, expiry = 1, num_sim = 1000, \
                                time_step = 500, vol = 0.2, rf = 0.05)
    result = sim.get_result()
    # change type to dataframe
    result = pd.DataFrame(result).T
    asian_call = monte_carlo.Sim_asian(sim_result = result, rf = 0.05, expiry = 1, \
                                   option_type = 'call', sample_window = 20, strike = 100)
    print '# ---------------------- Asian call option ---------------------- #'
    
    print 'Asian Call option price fixed AD: ', asian_call.discrete_fix_arith()
    print 'Asian Call option price fixed GD: ', asian_call.discrete_fix_geo()
    print 'Asian Call option price floating AD: ', asian_call.discrete_floating_arith()
    print 'Asian Call option price floating GD: ', asian_call.discrete_floating_geo()
    print 'Asian Call option price fixed AC: ', asian_call.continuous_fix_arith()
    print 'Asian Call option price fixed GC: ', asian_call.continuous_fix_geo()
    print 'Asian Call option price floating AC: ', asian_call.continuous_floating_arith()
    print 'Asian Call option price floating GC: ', asian_call.continuous_floating_geo()
    
    asian_put = monte_carlo.Sim_asian(sim_result = result, rf = 0.05, expiry = 1, \
                                   option_type = 'put', sample_window = 20, strike = 100)
    print ""
    print '# ---------------------- Asian put option ---------------------- #'
    
    print 'Asian Put option price fixed AD: ', asian_put.discrete_fix_arith()
    print 'Asian Put option price fixed GD: ', asian_put.discrete_fix_geo()
    print 'Asian Put option price floating AD: ', asian_put.discrete_floating_arith()
    print 'Asian Put option price floating GD: ', asian_put.discrete_floating_geo()
    print 'Asian Put option price fixed AC: ', asian_put.continuous_fix_arith()
    print 'Asian Put option price fixed GC: ', asian_put.continuous_fix_geo()
    print 'Asian Put option price floating AC: ', asian_put.continuous_floating_arith()
    print 'Asian Put option price floating GC: ', asian_put.continuous_floating_geo()
    
    # ---------------------------------------------------------------------------- #
    # ----------------------- Part II: Sampling time window ---------------------- #
    # ---------------------------------------------------------------------------- #
    sample_time_list = np.arange(5, 150, 1)
    sample_time_fix = []
    sample_time_floating = []
    for i in sample_time_list:
        asian_call = monte_carlo.Sim_asian(sim_result = result, rf = 0.05, expiry = 1, \
                                       option_type = 'call', sample_window = i, strike = 100)
        sample_time_fix.append(asian_call.discrete_fix_arith())
#        sample_time_floating.append(asian_call.discrete_floating_arith())
        
    plt.figure(figsize = (12, 8))
    mpl.rc('font',family='Times New Roman')
    plt.plot(sample_time_list, sample_time_fix, label = 'fixed strike')
#    plt.plot(sample_time_list, sample_time_floating, label = 'floating strike')
    plt.xlabel('Length of time window', fontsize = 16)
    plt.ylabel('Asian option price', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()
    
    # ---------------------------------------------------------------------------- #
    # ----------------------- Part III: number of time steps --------------------- #
    # ---------------------------------------------------------------------------- #
    time_step_list = np.hstack([np.arange(1, 30, 1), np.arange(30, 2000, 50)])
#    time_step_dis = []
    time_step_con = []
    for i in time_step_list:
        sim = simulation.Simulation(s0 = 100, expiry = 1, num_sim = 1000, \
                                    time_step = i, vol = 0.2, rf = 0.05)
        result = sim.get_result()
        result = pd.DataFrame(result).T
        asian_call = monte_carlo.Sim_asian(sim_result = result, rf = 0.05, expiry = 1, \
                                       option_type = 'call', sample_window = 10, strike = 100)

#        time_step_dis.append(asian_call.discrete_fix_arith())
        time_step_con.append(asian_call.continuous_fix_arith())
        
    plt.figure(figsize = (12, 8))
#    plt.plot(time_step_dis, label = 'fixed strike')
    plt.plot(time_step_con, label = 'floating strike')
    plt.xlabel('Number of time steps', fontsize = 16)
    plt.ylabel('Asian option price', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.show()
        
        