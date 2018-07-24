# -*- coding: utf-8 -*-
"""
# CQF Module 5 Exam Problem 2 (1)
@author: QUAN YUAN
"""
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

class Interpolation:
    def __init__(self, data):
        '''
        points: dict type with two keys x and y
        eg: {x: [1,2,3,4], 'y': [2,3,4,5]}
        also could be dataframe type
        '''
        self.data = data
        
        data_type = type(self.data)
        if data_type == dict:
            df = pd.DataFrame(self.data)
        elif data_type == pd.core.frame.DataFrame:
            df = self.data
        else:
            print "Please input the right type dict or DataFrame"
         # sort the x
        df = df.sort_values(by = 'x', ascending = True)
        self.df = df
            
    def __single_point(self, x_point, fun, rev_fun):
        '''
        x: sigle point
        eg: 1
        return: interpolation value
        '''
        for i in range(len(self.df)-1):
            point_i0 = self.df.iloc[i]
            point_i1 = self.df.iloc[i+1]
            x_i0 = point_i0['x']
            x_i1 = point_i1['x']
            y_i0 = point_i0['y']
            y_i1 = point_i1['y']
            
            if x_point >= x_i0 and x_point <= x_i1:
                x = (x_point - x_i0)*fun(y_i1)*1.0/(x_i1 - x_i0) + \
                          (x_i1 - x_point)*fun(y_i0)*1.0/(x_i1 - x_i0)
                return rev_fun(x)
        
    def get(self, x_list, fun, rev_fun):
        '''
        x: list
        eg: [1,2,3]
        return: interpolation value
        NOTICE: has to code like this since piecewise function like [lambda, lamda]
                is not applicable
        '''
        x_result = []
        for each_x in x_list:
            x_result.append(self.__single_point(each_x, fun, rev_fun))
        return x_result

def premium_leg(cs):
    # global: df_result, P
    sum_pl = 0
    for i in range(1, len(P)):
        sum_pl += cs*df_result[i]*P[i]*dt/2 + cs*df_result[i]*P[i-1]*dt/2
    return sum_pl

def default_leg():
    # global: df_result, PD
    sum_dl = 0
    for i in range(len(PD)):
#        print N*(1-RR)*df_result[i+1]*PD[i]
        sum_dl += N*(1-RR)*df_result[i+1]*PD[i]
    return sum_dl

def mtm(x):
    cs = x[0]
    return premium_leg(cs) - default_leg()

if __name__ == '__main__':
    # time delta
    dt = 0.25
    # nominal
    N = 1
    # recovery rate
    RR = 0.4
    
    # Interpolation
    lambda_dict = {'x': [0, 1, 2, 3], 'y': [0, 0.00995, 0.02087, 0.02579]}
    df_dict = {'x': [0, 1, 2, 3], 'y': [1, 0.97, 0.94, 0.92]}
    
    lambda_inter = Interpolation(data = lambda_dict)
    lambda_result = lambda_inter.get(x_list = np.arange(0,3.1,0.25), \
                                     fun = lambda x: x, rev_fun = lambda x: x)
    
    df_inter = Interpolation(data = df_dict)
    df_result = df_inter.get(x_list = np.arange(0,3.1,0.25), \
                             fun = lambda x: np.log(x), rev_fun = np.exp)
    
    # P
    P = []
    sum_lambda = 0
    for each_lambda in lambda_result:
        sum_lambda += each_lambda*dt
        P.append(np.exp(-sum_lambda))
    
    # PD
    PD = pd.Series(P).diff(-1).dropna()
    
    # minimize mtm
    credit_spread = fsolve(mtm, x0 = [0.009])
    print 'credit_spread equals to {0} bps'.format(credit_spread[0]*10000)
    
    
    
    
    
    
    
    