# -*- coding: utf-8 -*-
'''
Backtest and strategy evaluation
'''
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')

class Backtest:
    '''
    Python backtest tool only for singal underlying asset and only for research purpose
    also only for low or middle level frequency
    '''
    def __init__(self, stop_lose = -1, stop_profit = 1, init_account = 0, \
                 cost = 0, rf = 0.03):
        self.stop_lose = stop_lose
        self.stop_profit = stop_profit
        self.init_account = init_account
        # rate/fix amount
        self.cost = cost
        self.rf = rf
    def __adj_pl(self, pl):
        if pl >= self.stop_profit:
            return self.stop_profit
        elif pl <= self.stop_lose:
            return self.stop_lose
        else:
            return pl
            
    def run(self, data, visual = False):
        '''
        ###input###
        data: dataframe type;
              each column in data: "date", "price", "signal", "volumn";
              assume: open in the 1st time point and open/close in the next
                "date": datetime type or string type
                "price": means the open/close price (assume continuous)
                "signal": means the signal obtained before the corresponding price
                      1 means buy one unit, -1 means sell one unit, 
                      0 means close all the position, 
                      1 1 1 means hold one position constantly
                "volumn": 1 means one unit
        stop_lose: based on the pl(start: position open, end: position close)
                   if pl < stop_lose, close position
        stop_profit: based on the pl(start: position open, end: position close), 
                     if pl > stop_profit, close position
        ###output###
        result, dataframe type;
                "date", datetime type
                "price", float
                "signal", signal in everyday
                "pl", profit and lose everyday(use the information tomorrow considering jump)
                "cum_pl", cumulative pl from 1 to ?
                "drawback", drawback everyday
        '''
        # --------------------- core backtest -------------------- #
        # upward from min to max
        data = data.sort_values(by = 'date', ascending = True)
        # initial price
        init = data['price'].iloc[0]
        # calculate daily P&L rate(not log return)
        data['daily_PLrate1'] = data['price'].diff(1).fillna(0)*data['signal']*1.0/data['price'].shift(1).fillna(1)
        # adjusted P&L with stop loss and profit
        data['daily_PLrate'] = data['daily_PLrate1'].map(self.__adj_pl)
        # if stop Truem, else False
        data['judge_trade'] = (data['daily_PLrate'] != data['daily_PLrate1']).map(lambda x: int(2*x))
        # calculate which day we need pay transaction cost, change True, else False
        data['judge_cost'] = (data['signal'].shift(1).fillna(method = 'bfill') != \
                              data['signal']).map(lambda x: int(2*x))
        data['judge_cost'] = data['judge_cost'] + data['judge_trade']
        # calculate transaction cost
        if self.cost < 1:
            transaction_cost = data['volumn']*data['price']*self.cost
        else:
            transaction_cost = data['volumn']*self.cost
        data['Tcost'] = data['judge_cost']*transaction_cost        
        # calculate daily P&L amount(not log return, include cost)
        data['daily_PLamount'] = data['price'].shift(1).fillna(0)*data['daily_PLrate']
        data['daily_PLamount'] = data['daily_PLamount']*data['volumn'] - data['Tcost']
        # calculate cumulative amount
        data['cum_PLamount'] = np.cumsum(data['daily_PLamount'])
        # calculate account value
        if self.init_account == 0:
            data['account'] = init + data['cum_PLamount']
        else:
            data['account'] = self.init_account + data['cum_PLamount']
        # calculate net value curve rate(begin from 1)
        data['new_value'] = data['account']/init
        # calculate drawback
        drawback = np.zeros(len(data))
        for i in range(len(data['new_value'])):
            drawback[i] = data['new_value'].iloc[i] - max(data['new_value'].iloc[:i+1])
        data['drawback'] = drawback
        # --------------------- basic statistics -------------------- #
        # annualized std
        vol = data['daily_PLrate'].std()*np.sqrt(252)
        # total return
        return_total = data['new_value'].iloc[-1] - 1
        # annualized return
        return_ = (data['new_value'].iloc[-1] - 1)*252.0/len(data)
        # shape ratio
        sharpe = (return_ - self.rf)/vol
        # maxdrawback
        maxdrawback = min(drawback)
        # max profit and lose in one day
        max_profit = max(data['daily_PLrate'])
        max_lose = min(data['daily_PLrate'])
        # --------------------- visualization option -------------------- #
        if visual == True:
            if data['date'].dtype == str:
                data['date'] = data['date'].map(lambda x: \
                    dt.datetime.strptime(x, '%Y-%m-%d'))
            data.index = data['date']
            data['new_value'].plot(title = 'Net value curve')
            plt.xlabel('Time')
            plt.ylabel('Net value')
            plt.show()
            data['drawback'].plot(title = 'Drawback curve')
            plt.xlabel('Time')
            plt.ylabel('Drawback')
            plt.show()
            print 'Volatility(annualized): ', vol
            print 'Return(annualized): ', return_
            print 'Total return:', return_total
            print 'Sharpe Ratio: ', sharpe
            print 'Max Drawback: ', maxdrawback
            print 'Max profit in one day', max_profit
            print 'Max lose in one day', max_lose
        return data

        