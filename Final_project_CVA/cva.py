# -*- coding: utf-8 -*-
"""
CQF Final Project Exposure Estimation
@author: QUAN
"""
import pandas as pd
import numpy as np

class CVA_IRS:
    '''
    Cal MTM price of IRS at each tensor
    Cal Exposure at each tensor
    Cal CVA for IRS
    '''
    def __init__(self, T, dt, N, RR, fix, df, PD):
        # time to maturity, numerical
        self.T = T
        # time delta, numerical
        self.dt = dt
        # principal of IRS, numerical
        self.N = N
        # recovery rate, numerical
        self.RR = RR
        # fix rate
        self.fix = fix
        # forward discount factor, list/np.array/pd.Series
        self.df = np.array(df)
        # probability of default, list/np.array/pd.Series
        self.PD = np.array(PD)
        # tensor
        self.tensor = np.arange(0, self.T+0.0001, self.dt)
        # discount factor table
        DF_table = []
        for i in range(len(self.df)):
            df_0t = self.df[i]
            df_ti = np.append(np.zeros(i), self.df[i:]*1.0/df_0t)
            DF_table.append(df_ti)
        DF_table = pd.DataFrame(DF_table)
        DF_table.columns = list(self.tensor)
        DF_table.index = list(self.tensor)
        self.DF_table = DF_table

    def mtm(self, fwd):
        '''
        Input: fwd: forward rate(dataframe)
        Output: mtml: list
        '''
        fwd = fwd.iloc[:,1:].values
        mtml = []
        # minus 1 because the last MtM must be 0
        for i, each_fwd in zip(range(len(self.tensor) - 1), fwd):
            mtml.append(sum((each_fwd[i:] - self.fix)*self.N*self.dt*\
                            self.DF_table.iloc[i, (i+1):].values))
        mtml.append(0)
        return mtml
        
    def exposure(self, mtml):
        '''
        Input: fwd: forward rate
        Output: expo: list
        '''
        expo = [max(i, 0) for i in mtml]
        return expo
    
    def cva(self, expo):
        '''
        Input: expo: exposure
        Output: cva value
        '''
        perd_expo = np.array([(i+j)*1.0/2 for i,j in zip(expo[1:], expo[:-1])])
        DF = self.DF_table.iloc[0].values
        DF = np.array([(i+j)*1.0/2 for i, j in zip(DF[:-1], DF[1:])])
        return sum(perd_expo*DF*(1-self.RR)*self.PD)
    
    def p_cva(self, expo):
        '''
        Input: expo: exposure
        Output: cva value
        '''
        perd_expo = np.array([(i+j)*1.0/2 for i,j in zip(expo[1:], expo[:-1])])
        DF = self.DF_table.iloc[0].values
        DF = np.array([(i+j)*1.0/2 for i, j in zip(DF[:-1], DF[1:])])
        return perd_expo*DF*(1-self.RR)*self.PD
        