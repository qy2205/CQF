# -*- coding: utf-8 -*-
"""
CQF Final Project HJM
@author: QUAN YUAN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.optimize import curve_fit

class HJM_cov:
    def __init__(self, data):
        '''
        data: Forward rates data. type: dataframe
        '''
        try:
            data = data.drop('date', axis = 1)
        except KeyError:
            pass
        self.data = data
        
    def est(self, method, visual = False):
        '''
        method: 'diff', 'logdiff'. type: string
        '''
        # calculate difference
        if method == 'diff':
            df_diff = self.data.diff(1).dropna()
        elif method == 'logdiff':
            df_diff = self.data.apply(np.log).diff(1).dropna()
        else:
            print 'Invalid input'
            return False
        # calculate covariance matrix and annualizeds
        df_corr = df_diff.cov()*252.0/10000
        if visual == True:
            # Difference
            df_diff.plot(legend = False, figsize = (15,4))
            plt.xlabel('Time')
            plt.ylabel('Difference')
            plt.show()
            # Difference
#            try:
#                sns.pairplot(df_diff[[1,6,15]])
#                plt.title('')
#                plt.xlabel('Time')
#                plt.ylabel('Difference')
#                plt.show()
#            except:
#                pass
            # Covariance Matrix
#            df_corr.plot(legend = False, title = 'Covariance Matrix Measure')
#            plt.xlabel('Tenors')
#            plt.ylabel('Tenors')
#            plt.show()
            
        # return result
        return df_corr

class PCA:
    def __init__(self, data):
        '''
        data: Covariance data. type: dataframe
        '''
        try:
            self.data = data.values
        except:
            self.data = data
  
    def jacobi_eig(self, tol = 1.0e-9): # Jacobi method
        ''' lam,x = jacobi(a,tol = 1.0e-9).
            Solution of std. eigenvalue problem [a]{x} = lam{x}
            by Jacobi's method. Returns eigenvalues in vector {lam}
            and the eigenvectors as columns of matrix [x].
        '''
        def maxElem(a): # Find largest off-diag. element a[k,l]
            n = len(a)
            aMax = 0.0
            for i in range(n-1):
                for j in range(i+1,n):
                    if abs(a[i,j]) >= aMax:
                        aMax = abs(a[i,j])
                        k = i; l = j
            return aMax,k,l
      
        def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
            n = len(a)
            aDiff = a[l,l] - a[k,k]
            if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
            else:
                phi = aDiff/(2.0*a[k,l])
                t = 1.0/(abs(phi) + np.sqrt(phi**2 + 1.0))
                if phi < 0.0: t = -t
            c = 1.0/np.sqrt(t**2 + 1.0); s = t*c
            tau = s/(1.0 + c)
            temp = a[k,l]
            a[k,l] = 0.0
            a[k,k] = a[k,k] - t*temp
            a[l,l] = a[l,l] + t*temp
            for i in range(k):      # Case of i < k
                temp = a[i,k]
                a[i,k] = temp - s*(a[i,l] + tau*temp)
                a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
            for i in range(k+1,l):  # Case of k < i < l
                temp = a[k,i]
                a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
                a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
            for i in range(l+1,n):  # Case of i > l
                temp = a[k,i]
                a[k,i] = temp - s*(a[l,i] + tau*temp)
                a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
            for i in range(n):      # Update transformation matrix
                temp = p[i,k]
                p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
                p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
      
        n = len(self.data)
        maxRot = 5*(n**2)       # Set limit on number of rotations
        p = np.identity(n)*1.0     # Initialize transformation matrix
        for i in range(maxRot): # Jacobi rotation loop
            aMax,k,l = maxElem(self.data)
            if aMax < tol: return np.diagonal(self.data),p
            rotate(self.data,p,k,l)
        print 'Jacobi method did not converge'

    def numpy_eig(self):
        lam, x = np.linalg.eig(self.data)
        lam, x = lam.real, x.real
        return lam, x

class Reg:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def poly(self, n, visual = True):
        z1 = np.polyfit(self.x, self.y, deg = n, full = True)
        p1 = np.poly1d(z1[0])
        print z1[0]
        formula_list = [str(round(j,6))+'x^'+str(n-i) for i,j in zip(range(n), z1[0])]
        formula = reduce(lambda x,y: x + ' + ' + y, formula_list)
        print 'Fitting Formula: ', formula
        print 'Residuals: ', round(z1[1][0],6)
        
        if visual == True:
            xp = np.linspace(min(self.x), max(self.x), 200)
            plt.plot(self.x, self.y, '.', label = 'origin')
            plt.plot(xp, p1(xp), '-', label = 'fitted')
            plt.legend()
            plt.show()
        return p1
    
    def fit(self, fun, visual = True):
        '''
        fun(x, paras)
        '''
        popt, pcov = curve_fit(fun, self.x, self.y)
        y2 = [fun(i, *popt) for i in self.x]
        print 'Residuals: ', sum(((np.array(self.y) - y2)**2))/len(self.y)
        print 'Formula coef', popt
        # create function
        def p1(*para):
            def p2(xt):
                return fun(xt, *para)
            return p2
        if visual == True:
            xp = np.linspace(min(self.x), max(self.x), 200)
            yplot = [fun(i, *popt) for i in xp]
            plt.plot(self.x, self.y, '.', label = 'origin')
            plt.plot(xp, yplot, '-', label = 'fitted')
            plt.legend()
            plt.show()
        return p1(*popt)

def drift(tau, *vol):
    if tau == 0:
        return 0
    else:
        dtau = 0.01
        N = int(tau*1.0/dtau)
        dtau = tau*1.0/N
        
    # using trapezium rule to compute
    # sum of multi factors
    sum_M = 0
    for j in range(len(vol)):
        M = 0.5*vol[j](0)
        for i in range(1,N):
            # not adjusted by *0.5 because of 
            # repeating terms x1...xn-1 - see trapezoidal rule
            M += vol[j](i*dtau)
        M = M + 0.5*vol[j](tau)
        M = M*dtau
        # Vol_1 represents v_i(t,T) and 
        # M1 represents the result of numerical integration
        M = vol[j](tau)*M
        sum_M += M
    return sum_M

class MC:
    def __init__(self, num, step):
        self.num = num
        self.step = step
    
    # slow based on dataframe
    def hjm(self, tensor, f0, drift, vol):
        '''
        drift: list type
        vol: list type(each) [[vol1], [vol2], [vol3], ...]
        '''
        # for each tensor
        # simulation result, dynamic update
        sim = pd.DataFrame({0: f0}).T
        sim.columns = tensor
        # drift and vol table
        drift_vol = pd.DataFrame(vol, index = ['vol'+str(i[0]) for i in enumerate(vol)]).T
        drift_vol['drift'] = drift
        drift_vol.index = tensor
        drift_vol = drift_vol.T
        
        # random number list
        rand_df = pd.DataFrame(np.random.randn(self.num, len(vol)))
        rand_df.columns = ['rand'+str(i[0]) for i in enumerate(vol)]
        # only for test
        # rand_df = pd.read_excel('test.xlsx')
        
        # for each row
        for i in range(self.num):
            each_row = []
            # zip only for calculating slope
            for tensor_i, tensor_j in zip(tensor[:-1], tensor[1:]):
                dF = sim.iloc[i][tensor_j] - sim.iloc[i][tensor_i]
                slope = dF*1.0/(tensor_j - tensor_i)
                _drift = drift_vol[tensor_i].loc[['drift']].values[0]
                _vol = drift_vol[tensor_i].loc[drift_vol.index.drop('drift')].values
                rand_list = rand_df.iloc[i].values
                # SDE
                each_row.append(sim.iloc[i][tensor_i] + _drift*self.step + \
                                sum(_vol*rand_list)*np.sqrt(self.step) + \
                                slope*self.step)
            # for the last element
            _drift = drift_vol[tensor_j].loc[['drift']].values[0]
            _vol = drift_vol[tensor_j].loc[drift_vol.index.drop('drift')].values
            rand_list = rand_df.iloc[i].values
            each_row.append(sim.iloc[i][tensor_i] + _drift*self.step + \
                                sum(_vol*rand_list)*np.sqrt(self.step) + \
                                slope*self.step)
            # update to sim
            sim.loc[i+1] = each_row
        sim.index = [i*self.step for i in range(self.num+1)]
        return sim
    
    # much faster based on numpy array
    def new_hjm(self, tensor, f0, drift, vol):
        # tensor difference
        tensor_diff = np.diff(tensor)
        tensor_diff = np.append(tensor_diff, tensor_diff[-1])
        tensor_diff = np.matrix(tensor_diff)
        
        row, col = tensor_diff.shape
        mc_result = np.zeros(shape=(self.num + 1, col))
        mc_result[0,:] = f0
        rdn_numbers = np.random.randn(self.num, len(vol))
        sum_vol = np.dot(rdn_numbers, np.array(vol))        
        sum_vol = np.matrix(sum_vol)
        
        for n in range(1, self.num + 1):
            mc_result[n,:col-1] = mc_result[n-1, :col-1] + \
                        drift[0,:col-1]*self.step + \
                        sum_vol[n-1,:col-1]*np.sqrt(self.step) + \
                        ((np.matrix(mc_result[n-1,1:] - mc_result[n-1,:-1]))/(tensor_diff[0,:col-1]))*self.step
            mc_result[n,col-1] = mc_result[n-1, col-1] + \
                                drift[0, col-1]*self.step + \
                                sum_vol[n-1, col-1]*np.sqrt(self.step) + \
                                ((np.matrix(mc_result[n-1, col-1] - mc_result[n-1,col-2]))/(tensor_diff[0,col-1]))*self.step
        return mc_result

class Piecewise:
    def __init__(self):
        pass
    def __chunks(self, y, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(y), n):
            yield y[i:i + n]
    def new(self, y, n):
        new_y = [np.mean(each) for each in list(self.__chunks(y, n))]
        new_y = [val for val in new_y for i in range(n)]
        return new_y
    

        