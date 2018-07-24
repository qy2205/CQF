# -*- coding: utf-8 -*-
# -------------------------- CQF Module 3 Exam Code -------------------------- #
# -------------------------------- Apirl 2018 -------------------------------- #
# -------------------------------- QUAN YUAN  -------------------------------- #
# Description: Finite Difference method for European and American option
import numpy as np
import pandas as pd

class FD:
    def __init__(self, vol, rf, expiry, strike, option_type, etype, nas):
        # volatility
        self.vol = vol
        # interest rate
        self.rf = rf
        # expiry
        self.expiry = expiry
        # strike
        self.strike = strike
        # option type 'call' or 'put'
        self.option_type = option_type
        # exercise type
        self.etype = etype
        # number of asset steps
        self.nas = nas
        
    def price(self, visual = False):
        # delta price of asset
        ds = self.strike*2.0/self.nas
        # for stability
        dt = 0.9/self.nas/self.nas/self.vol/self.vol
        # number of time step
        nts = int(self.expiry*1.0/dt) + 1
        
        if self.option_type == 'call':
            flag = 1
        elif self.option_type == 'put':
            flag = -1
        
        # initalization
        # the underlying asset price
        s = np.zeros([self.nas])
        # option value in the later day 
        vold = np.zeros([self.nas])
        # option value in the former day
        vnew = np.zeros([self.nas])
        # array to store the results
        # col 1: asset price, col 2: time, col 3 new option price
        dummy = np.zeros([self.nas, 3])
        
        # initialization
        for i in range(self.nas):
            # define asset price
            s[i] = i*ds
            # the option payoff in T(expiry)
            vold[i] = max(flag*(s[i] - self.strike), 0)
            dummy[i, 0] = s[i]
            dummy[i, 1] = vold[i]
        
        # for stroring option value in each time step
        ts_op = []
        ts_op.append(vold)
        
        # back time, from 1 because we have the option price in expiry
        for k in range(1, nts):
            # bottom and top are special
            for i in range(1, self.nas - 1):
                delta = (vold[i + 1] - vold[i - 1])*1.0/(2*ds)
                gamma = (vold[i + 1] - 2*vold[i] + vold[i - 1])*1.0/(ds**2)
                # get theta from bs equation
                theta = -0.5*self.vol*self.vol*s[i]*s[i]*gamma \
                        - self.rf*s[i]*delta + self.rf*vold[i]
                # the middle
                vnew[i] = vold[i] - theta*dt
            # the bottom (discount)
            vnew[0] = (1 - self.rf*dt)*vold[0]
            # the top
            vnew[self.nas - 1] = 2*vnew[self.nas - 2] - vnew[self.nas - 3]

            # for american option
            if self.etype == 'Y':
                for i in range(self.nas):
                    # early exercise
                    vold[i] = max(vnew[i], dummy[i, 1])
            elif self.etype == 'N':
                for i in range(self.nas):
                    # update the old
                    vold[i] = vnew[i]
            
            ts_op.append(vold)
            
        # for outputing
        dummy[:, 2] = vold
        dummy[:, 1] = [i*dt for i in range(self.nas)]
        dummy = pd.DataFrame(dummy, columns = ['asset', 'time', 'value'])
        
        # for visulization the solution surface
        if visual == True:
            # option value in s,t or just V(s, t)
            op_value = np.hstack(ts_op)
            # time step
            op_ts = np.hstack([[j*dt for i in range(self.nas)] for j in range(nts)])
            # asset step
            op_as = np.hstack([[i*ds for i in range(self.nas)] for j in range(nts)])
            dummy = pd.DataFrame({'asset': op_as, 'time': op_ts, \
                                  'value': op_value})
        return dummy
    
    
            
            