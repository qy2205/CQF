# Question 6
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns

# Question 4,5,6

class Var_backtest():
    
    def __init__(self, trade_data, observation, var_period):
        self.trade_data = trade_data
        self.observation = observation
        self.var_period = var_period
        
    def get_result(self):
        data = pd.read_excel(self.trade_data)
        # get log return
        data['log_return'] = data['Closing Price'].apply(np.log).diff()
        # get rolling std
        roll_std = [np.nan for i in range(self.observation)]
        for i, j in zip(range(1, len(data) - (self.observation - 1)), \
                        range((self.observation + 1), len(data)+1)):
            roll_std.append(np.std(data['log_return'][i:j], ddof = 1))
        data['roll_std'] = roll_std
        # calculate rolling var
        data['var'] = data['roll_std']*norm(0,1).ppf(0.01)*np.sqrt(self.var_period)
        
        return_10D = [np.nan for i in range(self.observation)]
        for i, j in zip(range((self.observation+1), len(data)-(self.var_period - 1)), \
                        range((self.observation+1) + (self.var_period - 1), len(data)+1)):
            return_10D.append(np.log(data['Closing Price'][j]) - \
                              np.log(data['Closing Price'][i]))
        return_10D += [np.nan for i in range(self.var_period)]
        data['return_10D'] = return_10D
        # backtest var, True = breach, False = not breach
        data['var_backtest'] = data['return_10D'] < data['var']
        data['var_backtest'] = data['var_backtest'].replace([True, False], ['1','0'])
        # plot var and 10D return time series
        sns.set_style("whitegrid")
        data[['var','return_10D']].plot(title = 'backtest var ' + str(self.observation), figsize = (12,6))
        # calculatethe percentage of VaR breaches
        prob_breach = sum(data['var_backtest'].map(int))*1.0/len(data[data['return_10D'] == data['return_10D']])
        print 'probability breach:', prob_breach
        return data
    
if __name__ == '__main__':
    var_backtest_21 = Var_backtest('FTSE100.xlsx', 21, 10)
    result_data = var_backtest_21.get_result()
#    result_data.to_excel('FTSE100_21.xlsx')
    var_backtest_42 = Var_backtest('FTSE100.xlsx', 42, 10)
    result_data = var_backtest_42.get_result()
#    result_data.to_excel('FTSE100_42.xlsx')