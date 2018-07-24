# -*- coding: utf-8 -*-
"""
evaluation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors 
import matplotlib as mpl
import copy

class Distribution:
    '''
    class for distribution analysis
    Input：Series like data
    '''
    
    def __init__(self, data):
        self.data = data
        
    def analysis(self, qqplot = True):
        print "# ================= DESCRIPTION OF DATA ================= #"
        print pd.DataFrame(self.data).describe()
        print ""
        print "# ==================== DISTRIBUTION ===================== #"
        plt.figure()
        mpl.rc('font',family='Times New Roman')
        sns.distplot(self.data)
        plt.xlabel('The range of data', fontsize = 16)
        plt.ylabel('Freqency', fontsize = 16)
        plt.title('Distribution', fontsize = 16)
        print "# ==================== NORMAL TEST ====================== #"
        self.norm_test()
        print "# ============= DESCRIPTION OF DISTRIBUTION ============= #"
        print "SKEWNESS: ", round(pd.Series(self.data).skew(), 4)
        print "KURTOSIS: ", round(pd.Series(self.data).kurt(), 4)
#        print "# ======================= QQ PLOT ======================= #"
        if qqplot == True:
            print "QQ-PLOT"
            self.qq_plot()
    
    def qq_plot(self):
        
        S_mean = np.mean(self.data)
        S_std = np.std(self.data)
        
        S_return = pd.DataFrame(self.data)
        S_return_norm = (S_return - S_mean)*1.0/S_std
        S_return_norm.columns = ['return_norm']
        S_return_norm_sort = S_return_norm.sort_values(by = 'return_norm')
        S_return_norm_sort.index = range(len(S_return_norm_sort))
        S_return_norm_sort['percentage'] = [(i+1)*1.0/len(S_return_norm_sort) for i in range(len(S_return_norm_sort))]
        S_return_norm_sort['norm'] = S_return_norm_sort['percentage'].map(stats.norm(0,1).ppf)
        x = S_return_norm_sort.iloc[10:-10]['return_norm']
        y = S_return_norm_sort.iloc[10:-10]['norm']
        
        plt.figure()
        
        plt.scatter(x, y, marker = ".")
        plt.scatter(x, x, marker = ".")
        plt.xlabel('Theoretical Quantile', fontsize = 16)
        plt.ylabel('Sample Quantile', fontsize = 16)
        plt.title('QQ plot', fontsize = 16)
        
    def norm_test(self):
        # D'Agostino-Pearson Test, sample size 20-50
        if 20 < len(self.data) <= 50:
            p_value = stats.normaltest(self.data)[1]
            name = 'normaltest'
        
        elif len(self.data) <= 20:
            p_value = stats.shapiro(self.data)[1]
            name = 'shapiro'
            
        elif 300 >= len(self.data) >= 50:
            # Hubert Lilliefors
            p_value = lilliefors(self.data)
            name = 'lillifors'
            
        elif len(self.data) > 300:
            p_value = stats.kstest(self.data, 'norm')[1]
            name = 'KSTEST'
            
        if p_value < 0.05:
            print "USE ", name
            print "DATA ARE NOT NORMALLY DISTRIBUTED"
            return False
        else:
            print "USE ", name
            print "DATA ARE NORMALLY DISTRIBUTED"
            return True
        
class Signal_est:
    '''
    the length of time between two signals (not ready)
    the signal frequency in one month and one year 
    the proportion of long/short signal
    '''
    def __init__(self, data):
        '''Input Series like data, index must be date (datetime type)'''
        self.data = data
        data1 = copy.deepcopy(data)
        data1['date'] = data1.index
        data1['month'] = data1['date'].map(lambda x: str(x)[:7])
        data1['year'] = data1['date'].map(lambda x: str(x)[:4])
        data1['signal'] = (data1['signal'] != data1['signal'].shift(1).fillna(0)).map(int)
        self.data1 = data1
    def analysis(self):
        length = len(self.data)
        # long
        long_rate = len(self.data[self.data['signal'] == 1])*1.0/length
        # short
        short_rate = len(self.data[self.data['signal'] == -1])*1.0/length
        # zero
        zero_rate = len(self.data[self.data['signal'] == 0])*1.0/length
        # moth and year freq
        
        summary_year = self.data1.groupby('year').sum()
        summary_month = self.data1.groupby('month').sum()
        
        print 'The Length of backtest time period is:', length, 'day'
        print 'The frequency of long signal:', round(long_rate, 4)*100, '%'
        print 'The frequency of short signal:', round(short_rate, 4)*100, '%'
        print 'The frequency of no signal:', round(zero_rate, 4)*100, '%'
        print 'The signal freq in year: '
        print summary_year
        print 'The signal freq in month: '
        print summary_month
        
class Risk_est:
    '''
    Input: DataFrame type, date, pl , signal, net_value, drawback
    Q-Q plot
    distribution normal test
    99% Value at Risk (VaR)
    99% Expected Shortfall (ES)
    the mean, standard deviation, kurtosis and skewness
    '''
    
    def __init__(self, data):
        self.data = data
    
    def analysis(self, visual = True):
        
        profit = sorted(self.data['daily_PLrate'])
        var = round(profit[int(len(profit)*0.01)], 6)
        
        es = []
        for i in range(int(len(profit)*0.01)):
            es.append(profit[i])
        es = round(np.mean(es),6)
        std = round(np.std(profit)*np.sqrt(250),6)
        mean = round(np.mean(profit),6)
        skew = round(pd.DataFrame(profit).skew().values[0], 6)
        excess_kurt = round(pd.DataFrame(profit).kurt().values[0] - 3, 6)
        
        dist_return = self.data['daily_PLrate'][self.data['daily_PLrate'] != 0]
        Return_dist = Distribution(dist_return)
        Return_dist.analysis()
        Return_dist.norm_test()
        
        if visual == True:
            print 'VAR:',var
            print 'Expected shortfall',es
            print 'Mean', mean
            print 'Standard deviation', std
            print 'Skew', skew
            print 'Excess Kurt', excess_kurt
            
class Profit_loss:
    
    def __init__(self, data):
        self.data = data
        
    def analysis(self, visual = True):
        
        data = self.data
        data['year'] = data['date'].map(lambda x: str(x)[:4])
        data['month'] = data['date'].map(lambda x: str(x)[:7])
#        data['week'] = data['date'].map(lambda x: str(x)[:4] + x.strftime("%W"))
        
#        data_week = data[['delta_net', 'week']]
        data_month = data[['daily_PLrate', 'month']]
        data_year = data[['daily_PLrate', 'year']]
        
#        data_week = data_week.groupby('week').sum()
        data_month = data_month.groupby('month').sum()
        data_year = data_year.groupby('year').sum()
        
        max_day = max(data['daily_PLrate'])
        min_day = min(data['daily_PLrate'])
        
        print 'The max profit in one day is ', \
        data[data['daily_PLrate'] == max_day].index[0], max_day
        print 'The max loss in one day is ', \
        data[data['daily_PLrate'] == min_day].index[0], min_day
        
        max_month = max(data_month['daily_PLrate'])
        min_month = min(data_month['daily_PLrate'])
        
        print 'The max profit in one month is ', \
        data_month[data_month['daily_PLrate'] == max_month].index[0], max_month
        print 'The max loss in one month is ', \
        data_month[data_month['daily_PLrate'] == min_month].index[0], min_month
        
        max_year = max(data_year['daily_PLrate'])
        min_year = min(data_year['daily_PLrate'])
        
        print 'The best year is ', \
        data_year[data_year['daily_PLrate'] == max_year].index[0], max_year
        print 'The worst year is ', \
        data_year[data_year['daily_PLrate'] == min_year].index[0], min_year
        
        # plot the distribution for daily return
        plt.figure(figsize = (12, 8))
        mpl.rc('font',family = 'Times New Roman')
        plt.title('Distribution for the daily return', fontsize = 16)
        sns.distplot(data['daily_PLrate'])
        plt.xlabel('Profit and Loss', fontsize = 16)
        plt.ylabel('Frequency', fontsize = 16)
        plt.show()
        
        # plot the distribution for monthly return
        plt.figure(figsize = (12, 8))
        mpl.rc('font',family = 'Times New Roman')
        plt.title('Distribution for the monthly return', fontsize = 16)
        sns.distplot(data_month['daily_PLrate'])
        plt.xlabel('Profit and Loss', fontsize = 16)
        plt.ylabel('Frequency', fontsize = 16)
        plt.show()
        
        # print the year data
        print data_year['daily_PLrate']

def get_max_drawdown_underwater(underwater):
    """Determines peak, valley, and recovery dates given an 'underwater'
    DataFrame.
    An underwater DataFrame is a DataFrame that has precomputed
    rolling drawdown.
    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.
    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """

    valley = np.argmin(underwater)  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery

def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    returns = returns.copy()
    df_cum = returns.cumsum()
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for t in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if (len(returns) == 0) or (len(underwater) == 0):
            break

    return drawdowns

def gen_drawdown_table(returns, top=10):
    """
    Places top drawdowns in a table.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """

    df_cum = returns.cumsum()
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(top)),
                                columns=['net drawdown in %',
                                         'peak date',
                                         'valley date',
                                         'recovery date',
                                         'duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'duration'] = len(pd.date_range(peak,
                                                                recovery,
                                                                freq='B'))
        df_drawdowns.loc[i, 'peak date'] = (peak.to_pydatetime()
                                            .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'valley date'] = (valley.to_pydatetime()
                                              .strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'net drawdown in %'] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100

    df_drawdowns['peak date'] = pd.to_datetime(df_drawdowns['peak date'])
    df_drawdowns['valley date'] = pd.to_datetime(df_drawdowns['valley date'])
    df_drawdowns['recovery date'] = pd.to_datetime(
        df_drawdowns['recovery date'])

    return df_drawdowns

def plot_drawdown_periods(returns, top=10, k=None, ax=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

#     y_axis_formatter = FuncFormatter(utils.one_dec_places)
#     ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    if k != None:
        df_cum_rets = returns.cumsum() + k
    else:
        df_cum_rets = returns.cumsum()
    df_drawdowns = gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['peak date', 'recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])

    ax.set_title('Top %i Drawdown Periods' % top)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Portfolio'], loc='upper left')
    ax.set_xlabel('')
    return ax
