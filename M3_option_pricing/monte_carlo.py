# -*- coding: utf-8 -*-
# -------------------------- CQF Module 3 Exam Code -------------------------- #
# -------------------------------- Apirl 2018 -------------------------------- #
# -------------------------------- QUAN YUAN  -------------------------------- #
# Description: Monte Carlo for simulated price
import numpy as np

class Sim_european:
    def __init__(self, sim_result, strike, rf, expiry, option_type):
        # sim_result type: dataframe
        # col: numbers of simulation
        # index: time_step
        self.sim_result = sim_result
        # option strike price
        self.strike = strike
        # option type 'call' or 'put'
        self.option_type = option_type
        # risk-free rate
        self.rf = rf
        # expiry
        self.expiry = expiry    
        
    def get_value(self):
        if self.option_type == 'call':
            s = self.sim_result.iloc[-1, :]
            payoff = s.apply(lambda x: max(x - self.strike, 0))
            value = payoff.apply(lambda x: x*np.exp(-self.rf*self.expiry)).mean()
            return value
        elif self.option_type == 'put':
            s = self.sim_result.iloc[-1, :]
            payoff = s.apply(lambda x: max(self.strike - x, 0))
            value = payoff.apply(lambda x: x*np.exp(-self.rf*self.expiry)).mean()
            return value
        
class Sim_asian:
    def __init__(self, sim_result, rf, expiry, option_type, \
                 sample_window = None, strike = None):
        # sim_result type: dataframe
        # col: numbers of simulation
        # index: time_step
        self.sim_result = sim_result
        # option type 'call' or 'put'
        self.option_type = option_type
        # risk-free rate
        self.rf = rf
        # expiry
        self.expiry = expiry
        # None value for default since floating circumstance
        self.strike = strike
        # sample_window for discrete sampling
        self.sample_window = sample_window
        
    def discrete_fix_arith(self):
        # calculate option value with discrete sampling
        # fixed strike and arithmetic average
        time_step = len(self.sim_result)
        sample_index = [i for i in range(time_step) if i%self.sample_window == 0]
        sample_data = self.sim_result.iloc[sample_index]
        A = sample_data.mean()
        if self.option_type == 'call':
            payoff = A.apply(lambda x: max(x - self.strike, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = A.apply(lambda x: max(self.strike - x, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        
    def discrete_fix_geo(self):
        # calculate option value with discrete sampling
        # fixed strike and Geometric average
        time_step = len(self.sim_result)
        sample_index = [i for i in range(time_step) if i%self.sample_window == 0]
        sample_data = self.sim_result.iloc[sample_index]
        A = sample_data.apply(np.log).mean().apply(np.exp)
        if self.option_type == 'call':
            payoff = A.apply(lambda x: max(x - self.strike, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = A.apply(lambda x: max(self.strike - x, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        
    def discrete_floating_arith(self):
        # calculate option value with discrete sampling
        # floating strike and arithmetic average
        time_step = len(self.sim_result)
        sample_index = [i for i in range(time_step) if i%self.sample_window == 0]
        sample_data = self.sim_result.iloc[sample_index]
        A = sample_data.mean()
        if self.option_type == 'call':
            payoff = (A - sample_data.iloc[-1,:]).apply(lambda x: max(x, 0))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = (sample_data.iloc[-1,:] - A).apply(lambda x: max(x, 0))
            return payoff.mean()
        
    def discrete_floating_geo(self):
        # calculate option value with discrete sampling
        # floating strike and Geometric average
        time_step = len(self.sim_result)
        sample_index = [i for i in range(time_step) if i%self.sample_window == 0]
        sample_data = self.sim_result.iloc[sample_index]
        A = sample_data.apply(np.log).mean().apply(np.exp)
        if self.option_type == 'call':
            payoff = (A - sample_data.iloc[-1,:]).apply(lambda x: max(x, 0))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = (sample_data.iloc[-1,:] - A).apply(lambda x: max(x, 0))
            return payoff.mean()
        
    def continuous_fix_arith(self):
        # calculate option value with continuous sampling
        # fixed strike and arithmetic average
        A = self.sim_result.mean()
        if self.option_type == 'call':
            payoff = A.apply(lambda x: max(x - self.strike, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = A.apply(lambda x: max(self.strike - x, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        
    def continuous_fix_geo(self):
        # calculate option value with continuous sampling
        # fixed strike and Geometric average
        A = self.sim_result.apply(np.log).mean().apply(np.exp)
        if self.option_type == 'call':
            payoff = A.apply(lambda x: max(x - self.strike, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = A.apply(lambda x: max(self.strike - x, 0)*\
                             np.exp(-self.rf*self.expiry))
            return payoff.mean()
        
    def continuous_floating_arith(self):
        # calculate option value with continuous sampling
        # floating strike and arithmetic average
        A = self.sim_result.mean()
        if self.option_type == 'call':
            payoff = (A - self.sim_result.iloc[-1,:]).apply(lambda x: max(x, 0))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = (self.sim_result.iloc[-1,:] - A).apply(lambda x: max(x, 0))
            return payoff.mean()

    def continuous_floating_geo(self):
        # calculate option value with continuous sampling
        # fixed strike and Geometric average
        A = self.sim_result.apply(np.log).mean().apply(np.exp)
        if self.option_type == 'call':
            payoff = (A - self.sim_result.iloc[-1,:]).apply(lambda x: max(x, 0))
            return payoff.mean()
        elif self.option_type == 'put':
            payoff = (self.sim_result.iloc[-1,:] - A).apply(lambda x: max(x, 0))
            return payoff.mean()