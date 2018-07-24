import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

def portfolio_opt(mu, sigma, corr, target, rf):
    S = np.diag(sigma)
    Sigma = np.dot(np.dot(S, corr), S)
    Sigma_inv = np.mat(Sigma).I
    lam = (target - rf)*1.0/np.dot(np.dot((mu - rf).T, Sigma_inv), mu - rf)
    w = lam*np.dot(Sigma_inv, (mu - rf))
    return w, Sigma

if __name__ == '__main__':
    mu = np.array([0.04, 0.08, 0.12, 0.15])
    sigma = np.array([0.07, 0.12, 0.18, 0.26])
    corr = np.array([[1,0.2,0.5,0.3], [0.2,1,0.7,0.4],[0.5,0.7,1,0.9],[0.3,0.4,0.9,1]])
    target = [0.05, 0.075, 0.1, 0.125]
    rf = 0.03
    # calculate w, sigma, mu, in a range of target return values 5%, 7.5%, 10%, 12.5%
    for each in target:
        print 'when target return equals to', each
        w, Sigma = portfolio_opt(mu, sigma, corr, each, rf)
        print 'weight: ', w
        sigma_II = np.sqrt(np.dot(np.dot(w,Sigma), w.T))
        print 'sigma_II: ', sigma_II
        mu_II = np.dot(w, mu)
        print 'mu_II: ',mu_II
        print ""
        
    # plot the efficient froniter
    target_list = np.arange(0, 0.126, 0.001)
    sigma_II_list = []
    mu_II_list = []
    for each in target_list:
        w, Sigma = portfolio_opt(mu, sigma, corr, each, rf)
        sigma_II_list.append(float(np.sqrt(np.dot(np.dot(w,Sigma), w.T))))
        mu_II_list.append(float(np.dot(w, mu)))
    
    sns.set_style("whitegrid")
    plt.figure(figsize = (12, 8))
    plt.plot(sigma_II_list, mu_II_list)
    plt.xlabel('sigma')
    plt.ylabel('mu')
    plt.title('the efficient frontier')
    