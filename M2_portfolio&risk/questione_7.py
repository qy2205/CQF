import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def qq_plot(S_return):
    sns.set_style("whitegrid")
    S_mean = np.mean(S_return)
    S_std = np.std(S_return)
    S_return = pd.DataFrame(S_return)
    S_return_norm = (S_return - S_mean)*1.0/S_std
    S_return_norm.columns = ['return_norm']
    S_return_norm_sort = S_return_norm.sort_values(by = 'return_norm')
    S_return_norm_sort.index = range(len(S_return_norm_sort))
    S_return_norm_sort['percentage'] = [(i+1)*1.0/len(S_return_norm_sort) for i in range(len(S_return_norm_sort))]
    S_return_norm_sort['norm'] = S_return_norm_sort['percentage'].map(stats.norm(0,1).ppf)
    x = S_return_norm_sort.iloc[10:-10]['return_norm']
    y = S_return_norm_sort.iloc[10:-10]['norm']
    plt.figure(figsize=(12,8))
    plt.scatter(x, y, marker = ".")
    plt.scatter(x, x, marker = ".")


if __name__ == '__main__':
    ftse100 = pd.read_excel('FTSE100.xlsx')
    ftse100['log_return_1D'] = ftse100['Closing Price'].apply(np.log).diff(1)
    ftse100['log_return_10D'] = ftse100['Closing Price'].apply(np.log).diff(10)
    qq_plot(ftse100['log_return_1D'])
    qq_plot(ftse100['log_return_10D'])