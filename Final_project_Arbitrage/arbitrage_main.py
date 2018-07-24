# -*- coding: utf-8 -*-
"""
ARBITRAGE MAIN
"""
import pandas as pd
import numpy as np
import copy
import evaluation
import arbitrage
import backtest
import gridsearch

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns

from sklearn import linear_model

# plot setting
params = {'legend.fontsize': 12,
          'figure.figsize': (14, 6),
         'axes.labelsize': 18,
         'axes.titlesize': 18,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14,
         'font.family': 'Times New Roman'}
plt.rcParams.update(params)

# ---------------------------------------------------------------------------- #
# ----------------------------- Data Preparation ----------------------------- #
# ---------------------------------------------------------------------------- #
file_name = ['AMID', 'CLR', 'COMEX_AG', 'COMEX_AU', 'CVX', 'HES', 'IMO', \
             'NYMEX_CRUDE', 'NYMEX_GAS', 'OXY', 'RDS_B', 'USDX', 'XOM']
df_list = []
for each_file in file_name:
    each_df = pd.read_excel('data/' + each_file + '.xlsx')
    df_list.append(each_df)
data = reduce(lambda x,y: pd.merge(x,y, on = 'date'), df_list)
# research data from 2015-01-01 to present
data = data[data['date'] >= '2015-01-01']

# visualization for price data
chg_data = copy.deepcopy(data)
for each_col in data.columns.drop('date'):
    chg_data[each_col] = data[each_col]/data[each_col].iloc[0]
chg_data.index = chg_data['date']
del chg_data['date']
chg_data.plot()
plt.ylabel('Adjusted Price')
plt.title('Time Series Plot for Trading Assets')
plt.show()

# ---------------------------------------------------------------------------- #
# ------------------------ Vector Autoregression Model ----------------------- #
# ---------------------------------------------------------------------------- #
return_data = chg_data.apply(np.log).diff().dropna()
# visualization for asset return(stationary time series)
#return_data.plot(subplots = True)
#plt.show()
return_data['date'] = return_data.index
# Vector Autoregression
VAR = arbitrage.Vector_autoreg(return_data)
var_result = [VAR.stats(i) for i in range(1,11)]
# AIC, BIC
aic = [each_var.aic for each_var in var_result]
bic = [each_var.bic for each_var in var_result]
print '=============== AIC&BIC Test For Optimal Lag ==============='
print pd.DataFrame({'Lag': range(1,11), 'AIC': aic, 'BIC': bic})
print 'The optimal lag is: ', aic.index(min(aic)) + 1
# Residual Corr
print '=========== Residual Correlation Matrix(Lag = 1) ==========='
print var_result[0].corr
# Coeff
print '=================== Coefficients(Lag = 1) =================='
print var_result[0].coefs
# Stable
print '====================== Stable(Lag = 1) ====================='
print var_result[0].stable
# Result Summary
print '====================== Summary(Lag = 1) ===================='
print var_result[0].summary

# ---------------------------------------------------------------------------- #
# ------------------- Engle-Granger procedure with CADF Test ----------------- #
# ---------------------------------------------------------------------------- #
# pairs
data.index = data['date'].values
del data['date']
EGTest = arbitrage.EG_ADF()
EGResult = []
# asset 1 x
for i in data.columns:
    each_eg = []
    # asset 2 y
    for j in data.columns:
        if i == j:
            each_eg.append(0)
            continue
        x = data[i]
        y = data[j]
        # E-G test
        each_eg.append(EGTest.test(x = x, y = y))
    EGResult.append(each_eg)
EGresult_df = pd.DataFrame(EGResult, index = data.columns, \
                           columns = data.columns)
EGresult_beta_df = EGresult_df.apply(lambda x: \
                      x.map(lambda x: 0 if x == 0 else x.beta[0][0]))
EGresult_dfstats_df = EGresult_df.apply(lambda x: \
                      x.map(lambda x: 0 if x == 0 else x.dfstat))
EGresult_dfp_df = EGresult_df.apply(lambda x: \
                      x.map(lambda x: 0 if x == 0 else x.dfp))
EGresult_alphat_df = EGresult_df.apply(lambda x: \
                      x.map(lambda x: 0 if x == 0 else x.alphat))
EGresult_alphacon_df = EGresult_df.apply(lambda x: \
                      x.map(lambda x: 0 if x == 0 else x.alphacon))
EGresult_alphacoef_df = EGresult_df.apply(lambda x: \
                      x.map(lambda x: 0 if x == 0 else x.alphacoef))
EGresult_alphap_df = EGresult_df.apply(lambda x: \
                      x.map(lambda x: 0 if x == 0 else x.alphap))

# ---------------------------------------------------------------------------- #
# -------------------------------- Kalman Filter ----------------------------- #
# ---------------------------------------------------------------------------- #
asset1 = ['amid', 'xom', 'imo', 'xom', 'xom', 'xom']
asset2 = ['gas', 'ag', 'usdx', 'au', 'gas', 'rds_b']
intern_std_list = []
slope_std_list = []
Robust = arbitrage.Kalman(data)
for each_asset1, each_asset2 in zip(asset1, asset2):
    kalman_result = Robust.analysis(each_asset1, each_asset2, visual = False)
    kalman_result['intercept'] = (kalman_result.intercept - \
                 kalman_result.intercept.min())/(kalman_result.intercept.max() - \
                 kalman_result.intercept.min())
    kalman_result['slope'] = (kalman_result.slope - \
                 kalman_result.slope.min())/(kalman_result.slope.max() - \
                 kalman_result.slope.min())
    intern_std_list.append(kalman_result.intercept.std())
    slope_std_list.append(kalman_result.slope.std())
intern_slope_df = pd.DataFrame({'inter': intern_std_list, \
                             'slope': slope_std_list})
# sample viusalization for AMID/GAS
#kalman_result = Robust.analysis(asset1[0], asset2[0], visual = True)
# XOM/AG
#kalman_result = Robust.analysis(asset1[1], asset2[1], visual = True)
# ---------------------------------------------------------------------------- #
# --------------------------------- OU Fitting ------------------------------- #
# ---------------------------------------------------------------------------- #
Strategy_OU = arbitrage.OU()
halflife_list = []
sigma_e_list = []
mu_list = []
spread_list = []
for each_asset1, each_asset2 in zip(asset1, asset2):
    asset_spread = EGresult_df.loc[each_asset1, each_asset2].et
    spread_list.append(asset_spread)
    OUResult = Strategy_OU.fit(asset_spread)
    halflife_list.append(OUResult.halflife[0][0])
    sigma_e_list.append(OUResult.sigma_e[0][0])
    mu_list.append(OUResult.mu[0][0])
OU_result_df = pd.DataFrame({'asset1': asset1, 'asset2': asset2, 'halflife': halflife_list, \
                             'sigma_e': sigma_e_list, 'mu': mu_list})

# ---------------------------------------------------------------------------- #
# -------------------------------- Visualization ----------------------------- #
# ---------------------------------------------------------------------------- #
plt.plot(spread_list[0], label = 'Spread')
select_asset1 = OU_result_df.iloc[0]['asset1']
select_asset2 = OU_result_df.iloc[0]['asset2']
select_mu = OU_result_df.iloc[0]['mu']
select_sigma = OU_result_df.iloc[0]['sigma_e']
up_bound = float(select_mu) + float(select_sigma)
low_bound = float(select_mu) - float(select_sigma)
longmean = float(select_mu)

plt.plot(asset_spread.index, [longmean]*len(asset_spread), label = 'mu')
plt.plot(asset_spread.index, [up_bound]*len(asset_spread))
plt.plot(asset_spread.index, [low_bound]*len(asset_spread))
plt.title(select_asset1.upper() + ' AND '+ select_asset2.upper() + ' SPREAD')
#plt.plot(point1, point2)
plt.xlabel('TIME')
plt.ylabel('SPREAD')
plt.legend()
plt.show()

# ---------------------------------------------------------------------------- #
# ------------------------------ Single Generation --------------------------- #
# ---------------------------------------------------------------------------- #
Signal = arbitrage.Pairsignal()
pairsignal = Signal.generate(spread = spread_list[0], mu = longmean, \
                up = up_bound, low = low_bound)
data_backtest = copy.deepcopy(pairsignal)
coin_beta = EGresult_df.loc['amid', 'gas'].beta[0][0]
data_backtest['price'] = data['gas'] - coin_beta*data['amid']
data_backtest['date'] = data_backtest.index
data_backtest['volumn'] = 1
# visualization
data_backtest['signal'].plot()
plt.title('Pair Trading Strategy Signal Result')
plt.xlabel('Time')
plt.ylabel('Spread')
plt.show()
# ---------------------------------------------------------------------------- #
# ------------------------------ Strategy Backtest --------------------------- #
# ---------------------------------------------------------------------------- #
Pairbacktest = backtest.Backtest(stop_lose = -1, stop_profit = 1)
backtest_result = Pairbacktest.run(data_backtest, visual = True)
# return distribution analysis
#dist_return = backtest_result['daily_PLrate'][backtest_result['daily_PLrate'] != 0]
#Return_dist = evaluation.Distribution(dist_return)
#Return_dist.analysis()
# signal analysis
print '====================== Signal Analysis ======================'
Pair_signal_est = evaluation.Signal_est(data = pairsignal)
Pair_signal_est.analysis()
# risk analysis
print '======================= Risk Analysis ======================='
Pair_risk_est = evaluation.Risk_est(data = backtest_result)
Pair_risk_est.analysis()
# drawdown analysis
print '======================= P&L Analysis ========================'
#evaluation.plot_drawdown_periods(backtest_result['daily_PLrate'])
#plt.show()
Pair_PL_est = evaluation.Profit_loss(data = backtest_result)
Pair_PL_est.analysis()
# factor analysis
print '==================== Factor Backtesting ====================='
# data process
FF3 = pd.read_csv('data/Fama-French3.CSV')
SP500 = pd.read_excel('data/SP500.xlsx')
SP500['date'] = SP500['date'].map(lambda x: str(x)[:10])
SP500['SP500'] = SP500['SP500'].map(np.log).diff(1).fillna(0)
FF3.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF']
FF3['date'] = FF3['date'].map(lambda x: str(x)[:4] + '-' + \
                                        str(x)[4:6] + '-' + str(x)[6:])
backtest_result_ff = backtest_result[['date', 'daily_PLrate']]
backtest_result_ff['date'] = backtest_result_ff['date'].map(lambda x: str(x)[:10])
FF_return = pd.merge(backtest_result_ff, FF3, on = 'date')
SP_FF_return = pd.merge(FF_return, SP500, on = 'date')
# backtesting
roll_SP = []
roll_SMB = []
roll_HML = []
for i in range(len(FF_return)):
    if i < 100:
        continue
    Y_Strategy = SP_FF_return.loc[:i,['daily_PLrate']]
    X_Factors  = SP_FF_return.loc[:i,['SMB', 'HML', 'SP500']]
    # Create linear regression object
    OLS = linear_model.LinearRegression(fit_intercept = True)
    # Train the model using the training sets
    ols_result = OLS.fit(X_Factors, Y_Strategy)
    ols_coef = ols_result.coef_[0]
    roll_SP.append(ols_coef[2])
    roll_SMB.append(ols_coef[0])
    roll_HML.append(ols_coef[1])
FB_result = pd.DataFrame({'date': FF_return['date'].iloc[100:],'SP': roll_SP, \
                          'SMB': roll_SMB, 'HML': roll_HML})
# visualization
FB_result[['SMB', 'HML']].plot()
plt.title('Factor backtesting for HML and SMB')
plt.xlabel('Time')
plt.ylabel('Coef')
FB_result[['SP']].plot()
plt.title('Factor backtesting for SP500')
plt.xlabel('Time')
plt.ylabel('Coef')
# ---------------------------------------------------------------------------- #
# ---------------------------- Strategy Optimization ------------------------- #
# ---------------------------------------------------------------------------- #
spread = spread_list[0]
mu = OU_result_df.iloc[0]['mu']
sigma = OU_result_df.iloc[0]['sigma_e']
beta = coin_beta
asset1 = 'amid'
asset2 = 'gas'
rf = 0.03

def pair_target(up_bound, low_bound):
    up_bound = mu + up_bound*select_sigma
    low_bound = mu - low_bound*select_sigma
    Signal = arbitrage.Pairsignal()
    pairsignal = Signal.generate(spread = spread, mu = mu, \
                    up = up_bound, low = low_bound)
    data_backtest1 = copy.deepcopy(pairsignal)
    data_backtest1['price'] = data[asset2] - beta*data[asset1]
    data_backtest1['date'] = data_backtest.index
    data_backtest1['volumn'] = 1
    Pairbacktest = backtest.Backtest(stop_lose = -1, stop_profit = 1)
    backtest_result1 = Pairbacktest.run(data_backtest1, visual = False)
    # annulized vol
    vol = backtest_result1['daily_PLrate'].std()*np.sqrt(252)
    # annualized return
    return_ = (backtest_result1['new_value'].iloc[-1] - 1)*252.0/len(backtest_result1)
    # shape ratio
    sharpe = (return_ - rf)*1.0/(vol+0.001)
    return sharpe

Opt_pair = gridsearch.Optimizer(pair_target)
opt_result = Opt_pair.gridsearch(np.arange(0.8,1.21,0.05), \
                                 np.arange(0.8,1.21,0.05))
# visualization
x, y = np.meshgrid(sorted(set(opt_result['low_bound'])), \
                   sorted(set(opt_result['up_bound'])))
z = np.reshape(opt_result['result'].values, x.shape)
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap = cm.YlGnBu_r)
ax.set_xlabel('up_bound', labelpad = 15)
ax.set_ylabel('low_bound', labelpad = 15)
ax.set_zlabel('sharpe ratio', labelpad = 15)
plt.show()


