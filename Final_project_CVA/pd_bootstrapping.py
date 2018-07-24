# -*- coding: utf-8 -*-
"""
CQF Final Project CDS Bootstrapping
@author: QUAN YUAN
"""
import pandas as pd
import numpy as np

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

def prob_survival(D, RR, dt, S, N):
    '''
    Input:
    D: discount factor
    RR: Recover Rate
    dt: delta t
    S: credit spread
    N: N-period
    
    Output:
    P(T1)——P(TN) list type
    '''
    P = np.zeros(N+1)
    L = 1 - RR
    if N == 0:
        P[0] = 1
    elif N == 1:
        P[0] = 1
        P[N] = L*1.0/(L + dt*S[0])
    else:
        P[0] = 1
        P[1] = L*1.0/(L + dt*S[0])
        count = 2
        
        while count <= N:
            # sigma part
            sigma = 0
            for i in range(1,count):
                sigma += D[i]*(L*P[i-1] - (L+dt*S[count-1])*P[i])
            P[count] = sigma*1.0/(D[count]*(L + dt*S[count-1])) + \
                       P[count-1]*L/(L + dt*S[count-1])
            count += 1
    return P

