# -*- coding: utf-8 -*-
"""
ARBITRAGE MODULE
"""
import numpy as np
import pandas as pd
# Visualization Package
import matplotlib.pyplot as plt
# VAR Package
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
# Linear Regression Package
from sklearn import linear_model
import statsmodels.api as sm
from pykalman import KalmanFilter
#import copy

#import backtest

class CRESULT:
    '''
    class for storage results 
    
    '''
    def __init__(self):
        pass

class Reg:
    '''
    class for OLS estimation
    '''
    def __init__(self):
        pass
        
    def my_ols(self, x, y):
        x = np.array(x)
        y = np.array(y)
        G = np.linalg.inv(np.dot(x.T, x))
        para = np.dot(G, np.dot(x.T, y))
        res_hat = y - np.dot(x, para)
        OLSresult = CRESULT()
        OLSresult.coef_ = para
        OLSresult.residual = res_hat
        return OLSresult
        
    def skl_ols(self, x, y):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        # Create linear regression object
        OLS = linear_model.LinearRegression()
        # Train the model using the training sets
        OLS.fit(x, y)
        coefs = OLS.coef_
        inter = OLS.intercept_
        OLSresult = CRESULT()
        OLSresult.coefs = coefs
        OLSresult.intercept = inter
        OLSresult.residual = y.values - (np.dot(x.values, coefs) + inter)
        return OLSresult
    def stats_ols(self, x, y):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        x = sm.add_constant(x)
        stats_OLS = sm.OLS(y, x).fit()
        OLSresult = CRESULT()
        OLSresult.t = pd.DataFrame(stats_OLS.tvalues, columns = ['T_value'])
        OLSresult.con =  stats_OLS.conf_int()
        OLSresult.beta =  stats_OLS.params
        OLSresult.p = stats_OLS.pvalues
        return OLSresult

class Vector_autoreg:
    '''
    Vector Autoregression Model
    '''
    def __init__(self, data):
        '''
        data: dataframe type, must have one column 'date'
        'date' type: str or datetime like
        data must be stationary based on VaR model assumptions
        '''
        if type(data['date'].iloc[0]) == str:
            data.index = dates_from_str(data['date'].values)
        else:
            data.index = data['date'].values
        del data['date']
        self.data = data
        
    def __dependent_matrix(self):
        pass
    def __explantory_matrix(self):
        pass
    def myvar(self, p):
        '''
        VaR Model code by myself
        '''
        pass
    def stats(self, p):
        '''
        VaR Model from statsmodel for testing
        p: lag
        '''
        Var_result = CRESULT()
        varmodel = VAR(self.data)
        results = varmodel.fit(p)
        Var_result.summary = results.summary()
        # AIC and BIC
        Var_result.aic = results.aic
        Var_result.bic = results.bic
        # Coefficient
        if p == 1:
            Var_result.coefs = pd.DataFrame(results.coefs[0], \
                                       index = self.data.columns, \
                                       columns = 'Lag_'+self.data.columns)
        else:    
            Var_result.coefs = results.coefs
        # Correlation
        Var_result.corr = pd.DataFrame(results.resid_corr, \
                                       index = self.data.columns, \
                                       columns = self.data.columns)
        # Stable
        eignval_list = [abs(np.linalg.eig(i)[0]) for i in results.coefs]
        eignval_df = pd.DataFrame(eignval_list).T
        eignval_df.columns = ['lag'+str(i) for i in range(1,p+1)]
        Var_result.stable = eignval_df
        return Var_result

class EG_ADF:
    '''
    Engle-Granger procedure with CADF Test
    Step1: calculate et(et = y - bx) Run Regression
    Step2: Test et for stationary with CADF
    Step3: delta_Pat = theta*delta_Pbt - (1 - alpha)*e_(t-1)
    
    index must be time
    '''
    def __init__(self):
        pass
    def test(self, x, y):
        x = pd.DataFrame(x)
        # time from past to most recent
        x = x.sort_index()
        y = pd.DataFrame(y)
        y = y.sort_index()
        EGresult = CRESULT()
        EGreg = Reg()
        # step1: calculate e_t
        red_result = EGreg.skl_ols(x = x, y = y)
        e_t = pd.DataFrame(red_result.residual.T[0], index = x.index)
        EGresult.et = e_t
        EGresult.beta = red_result.coefs
        EGresult.inter = red_result.intercept
        # step2: ADF Test
        dftest = adfuller(list(e_t.values.T[0]), autolag = 'AIC')
        # storage result
        EGresult.dfstat = dftest[0]
        EGresult.dfp = dftest[1]
        EGresult.critical = pd.DataFrame({'critical': dftest[4].keys(), \
                                          'statistic': dftest[4].values()})
        # step3: estimate whether 1 - alpha significant
        delta_y = y.diff(1).dropna()
        delta_x = x.diff(1).dropna()
        e_tm1 = e_t.shift(-1).dropna()
        delta_x['et'] = e_tm1.values
#        print delta_x.head()
#        print delta_y.head()
        alpha_result = EGreg.stats_ols(x = delta_x, y = delta_y)
        # storage result
        EGresult.alphat = alpha_result.t.loc['et'].values[0]
        EGresult.alphacon = alpha_result.con.loc['et'].values
        EGresult.alphacoef = alpha_result.beta['et'] + 1
        EGresult.alphap = alpha_result.p['et']
        return EGresult

class OU:
    '''
    Run regression: e(t) = C + Be(t-1) + epi
    theta = -lnB/tao
    mu(e) = C/(1-B)
    sigma_ou = sqrt(2theta*Sigma(epi)/(1-e^(2*theta*tao)))
    sigma_eq = sigma_ou/sqrt(2*theta)
    '''
    def __init__(self):
        pass
    def fit(self, data):
        '''
        data: et(stationary residual from linear regression)
        index must be date(time)
        '''
        data = pd.DataFrame(data)
        # time from past to most recent
        data = data.sort_index()
        x = data.iloc[:-1]
        y = data.iloc[1:]
#        print x.head()
#        print x.tail()
#        print y.head()
#        print y.tail()
        OUresult = CRESULT()
        OUreg = Reg()
        OUfit = OUreg.skl_ols(x, y)
        B = OUfit.coefs
        C = OUfit.intercept
        tau = 1.0/252
        red = OUfit.residual
        OUresult.theta = -np.log(B)/tau
        OUresult.mu = C*1.0/(1 - B)
        SSE = sum(red.T[0]**2)/len(red)
        OUresult.sigma_ou = np.sqrt(2*OUresult.theta*SSE/(1-np.exp(-2*OUresult.theta*tau)))
        OUresult.sigma_e = OUresult.sigma_ou/np.sqrt(2*OUresult.theta)
        OUresult.halflife = np.log(2)/(OUresult.theta*tau)
        return OUresult

class Kalman:
    '''
    Kalman filter for Regression Estimation
    Input: all dataset including two assets which we want to explore
    Output: Analysis Result
    Notice: index must be time format
    '''
    def __init__(self, data):
        self.data = data
    def analysis(self, asset1, asset2, visual = False):
        # Kalman Filter
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.vstack([self.data[asset1], \
                             np.ones(self.data[asset1].shape)]).T[:, np.newaxis]
        # set parameters
        kf = KalmanFilter(n_dim_obs = 1, n_dim_state = 2,
                          initial_state_mean=np.zeros(2),
                          initial_state_covariance = np.ones((2, 2)),
                          transition_matrices = np.eye(2),
                          observation_matrices = obs_mat,
                          observation_covariance = 1.0,
                          transition_covariance = trans_cov)
        # calculate rolling beta and intercept
        state_means, state_covs = kf.filter(self.data[asset2].values)
        beta_slope = pd.DataFrame(dict(slope=state_means[:, 0], \
                         intercept=state_means[:, 1]), index = self.data.index)
        if visual == True:
            # visualization for correlation
            cm = plt.cm.get_cmap('jet')
            colors = np.linspace(0.1, 1, len(self.data))
            sc = plt.scatter(self.data[asset1], self.data[asset2], \
                             s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
            cb = plt.colorbar(sc)
            cb.ax.set_yticklabels([str(p.date()) for p in \
                                   self.data[::len(self.data)//9].index]);
            plt.xlabel(asset1)
            plt.ylabel(asset2)
            plt.show()
           
            # plot beta and slope
            beta_slope.plot(subplots = True)
            plt.show()

            # visualize the correlation between assest prices over time
            cm = plt.cm.get_cmap('jet')
            colors = np.linspace(0.1, 1, len(self.data))
            sc = plt.scatter(self.data[asset1], self.data[asset2], \
                             s=50, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
            cb = plt.colorbar(sc)
            cb.ax.set_yticklabels([str(p.date()) for p in self.data[::len(self.data)//9].index]);
            plt.xlabel(asset1)
            plt.ylabel(asset2)
            
            # add regression lines
            step = 5
            xi = np.linspace(self.data[asset1].min(), self.data[asset1].max(), 2)
            colors_l = np.linspace(0.1, 1, len(state_means[::step]))
            for i, beta in enumerate(state_means[::step]):
                plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[i]))
        return beta_slope

class Pairsignal:
    def __init__(self):
        pass
    def generate(self, spread, mu, up, low):
        '''
        spread: dataframe object(index must be date/time)
        '''
        spread = spread.sort_index()
        spt = spread.values.T[0][1:]
        spt_1 = spread.values.T[0][:-1]
        signal = np.zeros(len(spread))
        signal[0] = np.nan
        for i in range(len(spt)):
            if spt[i] >= up and spt_1[i] < up:
                signal[i+1] = -1
            elif spt[i] <= low and spt_1[i] > low:
                signal[i+1] = 1
            elif (spt[i] >= mu and spt_1[i] < mu) or \
                 (spt[i] <= mu and spt_1[i] > mu):
                signal[i+1] = 0
            else:
                signal[i+1] = np.nan
        signal_df = pd.DataFrame(signal, columns = ['signal'], \
                                 index = spread.index).fillna(method = 'ffill').fillna(0)
        # shift only for backtesing
        return signal_df.shift(1).fillna(0)

    
    
    