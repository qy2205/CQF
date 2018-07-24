# -*- coding: utf-8 -*-
# -------------------------- CQF Module 3 Exam Code -------------------------- #
# -------------------------------- Apirl 2018 -------------------------------- #
# -------------------------------- QUAN YUAN  -------------------------------- #
# Description: Underlying stock simulation

import numpy as np

class Simulation:
    def __init__(self, s0, expiry, num_sim, time_step, vol, rf):
        # stock price at time 0
        self.s0 = s0
        # time to expriy
        self.expiry = expiry
        # the number of simulation
        self.num_sim = num_sim
        # the number of time step
        self.time_step = time_step
        # volatility
        self.vol = vol
        # risk free rate
        self.rf = rf
        
    def get_result(self):
        
        result = []
        delta_t = self.expiry*1.0/self.time_step
        
        # for each simulation
        for i in range(self.num_sim):
            # list fot store each sim result
            sim = []
            s_before = self.s0
            sim.append(s_before)
            # for each time step
            for j in range(self.time_step):
                s_after = s_before*(1 + self.rf*delta_t) + \
                    self.vol*s_before*np.sqrt(delta_t)*np.random.normal(0, 1)
                s_before = s_after
                sim.append(s_after)
            
            result.append(sim)
            
        return result
