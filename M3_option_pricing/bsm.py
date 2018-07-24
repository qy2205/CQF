# -*- coding: utf-8 -*-
# -------------------------- CQF Module 3 Exam Code -------------------------- #
# -------------------------------- Apirl 2018 -------------------------------- #
# -------------------------------- QUAN YUAN  -------------------------------- #
# Description: BSM for theoretical price

from scipy.stats import norm
import numpy as np

# class for theoretical price of European option
class Bsm_european:
    def __init__(self, s, strike, expiry, vol, rf):
        self.s = s
        self.strike = strike
        self.expiry = expiry
        self.vol = vol
        self.rf = rf
        
        # calculate d1 and d2
        d1 = (np.log(self.s*1.0/self.strike) + \
            (self.rf + 1.0*vol**2/2)*self.expiry)/(vol*np.sqrt(expiry))
        d2 = d1 - self.vol*self.expiry
        
        self.d1 = d1
        self.d2 = d2
        
    def call_value(self):
        call = self.s*norm.cdf(self.d1) - self.strike*\
                    np.exp(-self.rf*self.expiry)*norm.cdf(self.d2)
        return call
        
    def put_value(self):
        put = -self.s*(1 - norm.cdf(self.d1)) + self.strike*\
                    np.exp(-self.rf*self.expiry)*(1 - norm.cdf(self.d2))
        return put
    