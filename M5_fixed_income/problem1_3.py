# -*- coding: utf-8 -*-
"""
@author: QUAN YUAN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # the number of simulation
    num_sim = 2000
    # time to expriy
    T = 1 # year
    # the number of time step
    time_step = 252
    # risk free rate
    rf = 0.02
    # threshold
    k = 5
    # asset volatility
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # asset value
    v0_list = [7.901, 7.897, 7.855, 7.484, 7.586, 7.383, \
               7.156, 6.915]
    
    for sigma, v0 in zip(sigma_list, v0_list):
        result = []
        delta_t = T*1.0/time_step
        
        # for each simulation
        for i in range(num_sim):
            # list fot store each sim result
            sim = []
            v_before = v0
            sim.append(v_before)
            # for each time step
            for j in range(time_step):
                v_after = v_before*(1 + rf*delta_t) + \
                    sigma*v_before*np.sqrt(delta_t)*np.random.normal(0, 1)
                v_before = v_after
                sim.append(v_after)
            result.append(sim)
            
        result = pd.DataFrame(result).T
        result_min = result.min()
        PD = len(result_min[result_min < 5])*1.0/len(result_min)
        print 'Asset value is ', v0
        print 'Asset volatility is ', sigma
        print 'Probability of default: ', PD
    # =========================================================================
    #                       only for visualization(slow)
    # =========================================================================
    black_theo = pd.read_excel('black_theo.xlsx')
    sigma_list = black_theo['sigma']
    v0_list = black_theo['v0']
    
    PD_sim = []
    for sigma, v0 in zip(sigma_list, v0_list):
        result = []
        delta_t = T*1.0/time_step
        
        # for each simulation
        for i in range(num_sim):
            # list fot store each sim result
            sim = []
            v_before = v0
            sim.append(v_before)
            # for each time step
            for j in range(time_step):
                v_after = v_before*(1 + rf*delta_t) + \
                    sigma*v_before*np.sqrt(delta_t)*np.random.normal(0, 1)
                v_before = v_after
                sim.append(v_after)
            result.append(sim)
            
        result = pd.DataFrame(result).T
        result_min = result.min()
        PD = len(result_min[result_min < 5])*1.0/len(result_min)
        PD_sim.append(PD)
    
    plt.figure(figsize = (12, 8))
    plt.plot(v0_list, PD_sim)
    plt.plot(v0_list, black_theo['Theo_PD'])
    plt.title('Black-Cox Model Monte Carlo results VS Theoretical results', fontsize = 16)
    plt.xlabel('V0', fontsize = 16)
    plt.ylabel('Probability of default', fontsize = 16)
    plt.xlabel()
    plt.show()
    
    
    
    
    


