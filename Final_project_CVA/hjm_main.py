# -*- coding: utf-8 -*-
"""
CQF Final Project HJM Part
@author: QUAN YUAN
"""
import pandas as pd
import numpy as np
#from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import hjm

params = {'legend.fontsize': 12,
          'figure.figsize': (12, 6),
         'axes.labelsize': 18,
         'axes.titlesize': 18,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14,
         'font.family': 'Times New Roman'}
plt.rcParams.update(params)


# load data
blc = pd.read_excel('blcdata_2015_2018.xlsx')
# drop nan value
blc = blc.dropna()
# explore data and visualization (Time Series data with Fixed Tensor)
plt.figure()
plt.plot(blc['date'], blc[0.5], label = 'Tensor = 0.5')
plt.plot(blc['date'], blc[5], label = 'Tensor = 5')
plt.plot(blc['date'], blc[10], label = 'Tensor = 10')
plt.plot(blc['date'], blc[15], label = 'Tensor = 15')
plt.plot(blc['date'], blc[20], label = 'Tensor = 20')
plt.plot(blc['date'], blc[25], label = 'Tensor = 25')
plt.legend()
plt.xlabel('Historic Time, days')
plt.ylabel('Rates %')
plt.title('Time Series data with Fixed Tensor')
plt.show()
# Yield curve data with fixed time
plt.figure()
plt.plot(blc.columns.drop('date'), blc.loc[0][1:].values, \
         label = '2015-01-02', linewidth = 3)
plt.plot(blc.columns.drop('date'), blc.loc[107][1:].values, \
         label = '2015-06-02', linewidth = 3)
plt.plot(blc.columns.drop('date'), blc.loc[260][1:].values, \
         label = '2016-01-04', linewidth = 3)
plt.plot(blc.columns.drop('date'), blc.loc[367][1:].values, \
         label = '2016-06-01', linewidth = 3)
plt.plot(blc.columns.drop('date'), blc.loc[521][1:].values, \
         label = '2017-01-03', linewidth = 3)
plt.plot(blc.columns.drop('date'), blc.loc[628][1:].values, \
         label = '2017-06-01', linewidth = 3)
plt.plot(blc.columns.drop('date'), blc.loc[781][1:].values, \
         label = '2018-01-02', linewidth = 3)
plt.plot(blc.columns.drop('date'), blc.loc[889][1:].values, \
         label = '2018-06-01', linewidth = 3)
plt.legend()
plt.xlabel('Tensor Time')
plt.ylabel('Rates %')
plt.title('Forward Curves')
plt.show()

# =========================================================================== #
# ============================== PCA analysis =============================== #
# =========================================================================== #

# reduce the sample size to two years of stable curve regime
blc.index = blc['date']
stable_blc = blc[blc['date'] >= '2016-11-29']

# change the freq to weekly for robust cov
stable_blc.index = list(stable_blc.date)
del stable_blc['date']
week_blc = stable_blc.resample('W', kind='period').mean()

# calculate the cov matrix with class in hjm.py(using log difference)
Cov = hjm.HJM_cov(stable_blc)
blc_cov = Cov.est(method = 'diff', visual = True)

# matrix decomposition 
Pca = hjm.PCA(data = blc_cov)
lam, vec = Pca.jacobi_eig()

# variance attribution
sum_lam = sum(lam)
attr = lam/sum_lam

# to dataframe
eig_df = pd.DataFrame(vec).T
eig_df.columns = week_blc.columns
eig_df['lam'] = lam
eig_df['attr'] = attr
eig_df = eig_df.T

# top 5 pc
top_pc = sorted(eig_df.iloc[-1], reverse = True)[:5]
# plot attribution
plt.figure()
plt1 = plt.bar(range(len(top_pc)), top_pc, label = 'PCA for cov')
plt.title('Attribution of Top 5 Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Attribution %')
plt.legend()
for a,b in zip(range(len(top_pc)), top_pc):
    plt.text(a, b+0.006, '%.2f%%' %(b*100), ha='center', va= 'bottom', fontsize=16)
plt.show()

# get eigen vectors which corresponding to the top 4 highest eigenvalue()
# sum(12vec, 4vec, 50vec, 31vec) —— account for 98.88%
top4tenor = eig_df.loc[['attr'], :].sort_values(by = 'attr', \
          axis = 1, ascending = False).iloc[:,:4].columns
top4pc = eig_df[list(top4tenor)].iloc[:-2]
top4pc.columns = ['PC' + str(i) for i in range(1,5)]
# plot pc curve
top4pc.plot(title = 'PCA for Covariance matrix', \
            linewidth = 3, legend = True)
plt.xlabel('Tensor')
plt.ylabel('Egenvector, % movement at each tenor')
plt.ylim(-0.4, 0.4)
plt.grid()
plt.legend()
plt.show()

# =========================================================================== #
# =================================== HJM =================================== #
# =========================================================================== #
# calculate vol
lamsqrt_top4 = np.sqrt(np.array(sorted(lam, reverse=True)[:4]))
top4pc_m = top4pc.values
vol_top4 = lamsqrt_top4*top4pc_m
vol_df = pd.DataFrame(vol_top4)
vol_df.index = week_blc.columns
vol_df.columns = ['Vol1', 'Vol2', 'Vol3', 'Vol4']

# fitting vol curve with
# target: residual < 0.001
def fun1(x, a, b):
    return a*x**2 + b*np.log(x+0.000001)

# vol1
x = np.array(list(vol_df.index))
y1 = vol_df['Vol1'].values
Volfit1 = hjm.Reg(x = x, y = y1)
fvol1 = Volfit1.poly(n = 2, visual = True)
fvol1 = Volfit1.poly(n = 3, visual = True)
fvol1 = Volfit1.poly(n = 4, visual = True)
fvol1 = Volfit1.fit(fun1, visual = True)

# vol2
y2 = vol_df['Vol2'].values
Volfit2 = hjm.Reg(x = x, y = y2)
fvol2 = Volfit2.poly(n = 3, visual = True)

# vol3
y3 = vol_df['Vol3'].values
Volfit3 = hjm.Reg(x = x, y = y3)
fvol3 = Volfit3.poly(n = 3, visual = True)

# vol4
y4 = vol_df['Vol4'].values
Volfit4 = hjm.Reg(x = x, y = y4)
fvol4 = Volfit4.poly(n = 3, visual = True)

# calculate drift
drift_list = [hjm.drift(each_tau, fvol1, fvol2, fvol3, fvol4) \
              for each_tau in x]
# drift visualization
plt.plot(drift_list)
plt.title('HJM Drift Curve')
plt.xlabel('Tensor')
plt.ylabel('Drift')
plt.show()

# check
pd.DataFrame([fvol1(x), fvol2(x), fvol3(x), fvol4(x)]).to_csv('vol.csv')
pd.DataFrame(drift_list).to_csv('drift.csv')
pd.DataFrame(np.array(blc.drop('date', axis = 1).iloc[-1])/100).to_csv('f0.csv')

# Monte Carlo simulation for HJM
HJM_MC = hjm.MC(num = 1000, step = 0.01)
new_mc_result = HJM_MC.new_hjm(tensor = list(vol_df.index), \
                       f0 = np.array(blc.drop('date', axis = 1).iloc[-1])/100, \
                       drift = np.matrix(drift_list), \
                       vol = [fvol1(x), fvol2(x), fvol3(x), fvol4(x)])
mc_result = pd.DataFrame(new_mc_result, columns = vol_df.index)
# old method(slow) don't use
#mc_result = HJM_MC.hjm(tensor = list(vol_df.index), \
#                       f0 = np.array(blc.drop('date', axis = 1).iloc[-1])/100, \
#                       drift = drift_list, \
#                       vol = [fvol1(x), fvol2(x), fvol3(x), fvol4(x)])

# Visualization for Monte Carlo Result
mc_result.plot(legend = False, figsize = (15,6))
plt.title('Monte Carlo Simulation for HJM')
plt.xlabel('Future Historic Time, dt=0.01')
plt.ylabel('Forward rate')
plt.show()

sim_forward = mc_result.loc[[0, 100, 200, 500],:].T
sim_forward.columns = ['present', '1Y', '2Y', '5Y']
sim_forward.plot(title = 'Simulated Forward Curve')
plt.xlabel('Tenors')
plt.ylabel('Forward Rates')
plt.show()


# Visualization for Monte Carlo Result

# =========================================================================== #
# ================== Implied OIS Curve and Discount Factor ================== #
# =========================================================================== #
# load and clean data
libor_forward = pd.read_excel('liborforward.xlsx')
libor_forward.columns = ['date'] + map(lambda x: round(x, 2), \
                      list(libor_forward.columns)[1:])
libor_forward = libor_forward.dropna()
ois_forward = pd.read_excel('oisforward.xlsx')
ois_forward.columns = ['date'] + map(lambda x: round(x, 2), \
                    list(ois_forward.columns)[1:])
ois_forward = ois_forward.dropna()
# judge difference
print 'Judge: ', (ois_forward.index == libor_forward.index).all()

# calculate LOIS(time series)
lois = ((libor_forward.iloc[:,1:] - ois_forward.iloc[:,1:])/100)
lois.index = libor_forward['date']

# median for ts and also median for different tensors
med_lois = lois.median()

# visualization
lois.T.plot(legend = False, alpha = 0.3)
plt.plot(med_lois, 'r', linewidth = 3)
plt.xlabel('Time')
plt.ylabel('LOIS spread Curve')
plt.title('LOIS spread in Time Series')
plt.show()
# data process
x_tensor = np.array([med_lois.index])
y_lois = np.array([med_lois.values]).T
# boxing
Lois_box = hjm.Piecewise()
new_lois = Lois_box.new(y = y_lois, n = 10)
# visualization
plt.plot(x_tensor.T, new_lois, label = 'piecewise results', linewidth = 3)
plt.plot(x_tensor.T, y_lois, label = 'original data', linewidth = 3)
plt.title('Data boxing')
plt.legend()
plt.xlabel('Tensor')
plt.ylabel('LOIS')
plt.show()

# --------------------------------------------------------------------------- #
## research about LOIS CURVE Fitting

## different LOIS per tensor period with Kmeans++
#from sklearn.cluster import KMeans
#km_cluster = KMeans(n_clusters = 3, max_iter = 300, init = 'k-means++')
#km_result = km_cluster.fit_predict(y_lois)

## visualization for kmeans result
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(med_lois.values, '-r', label = 'LOIS Median', linewidth = 3)
#ax2 = ax.twinx()
#ax2.plot(km_result, label = 'Kmeans++ Results', linewidth = 3)
#ax.legend(loc = 0)
#ax.set_xlabel("Tensor")
#ax.set_ylabel("LOIS")
#ax2.set_ylabel("Clustering Results")
#ax2.legend(loc = 2)
# --------------------------------------------------------------------------- #
# calculate implied ois for all simulated forward libors(5 years)
lois_df = pd.DataFrame([new_lois for i in range(1001)])
lois_df.columns = x_tensor[0]
lois_df = lois_df[[0.08]+[i*0.5 for i in range(1,11)]]
# save lois for cva calculation
#pd.DataFrame(lois_df.loc[0]).T.iloc[:,1:].to_csv('lois.csv')
mc_result_5 = mc_result[[0.08]+[i*0.5 for i in range(1,11)]]
final_implied_ois = mc_result_5.iloc[0].values - lois_df.iloc[0].values
#implied_ois = mc_result_5 - lois_df
#final_implied_ois = implied_ois.mean()

# visualization
final_implied_ois_df = pd.DataFrame(final_implied_ois, columns = ['implied OIS'])
final_implied_ois_df.index = [0.08]+[i*0.5 for i in range(1,11)]
print '===================== implied OIS ====================='
print final_implied_ois_df
final_implied_ois_df.plot(figsize = (4,6))
plt.title('Implied OIS')
plt.xlabel('Tenors')
plt.ylabel('Rates')
plt.show()

# calculate implied discounting factor
implied_DF = [1]
delta_t = np.insert(np.diff(final_implied_ois_df.index), 0, 0.08)
DF = 1
for i,j in zip(delta_t, final_implied_ois):
    DF = np.exp(-i*j)*DF
    implied_DF.append(DF)
DF_df = pd.DataFrame({'tensor': [0, 0.08]+[i*0.5 for i in range(1,11)], \
                      'DF': implied_DF})
# above calculate the discount factor L(0, Ti)
# discounting table for DF L(Ti, Ti+1) for appropriately discounted
# don't use 0.08
DF_df = DF_df[DF_df['tensor'] != 0.08]
DF_table = []
for i in range(len(DF_df)):
    df_0t = DF_df.DF.iloc[i]
    df_ti = np.append(np.zeros(i), (DF_df.DF.iloc[i:]*1.0/df_0t).values)
    DF_table.append(df_ti)
DF_table = pd.DataFrame(DF_table)
DF_table.columns = list(DF_df.tensor)
DF_table.index = list(DF_df.tensor)
print DF_table
# save results
#DF_df.to_csv('DF.csv')
#mc_result.to_csv('mc_result.csv')
DF_table.to_csv('DF_table.csv')



