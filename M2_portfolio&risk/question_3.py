import numpy as np
from scipy.stats import norm

def tangency_portfolio_opt(mu, sigma, corr, rf):
    S = np.diag(sigma)
    Sigma = np.dot(np.dot(S, corr), S)
    Sigma_inv = np.mat(Sigma).I
    dim = np.shape(Sigma_inv)[0]
    A = np.dot(np.dot(np.ones((dim,)), Sigma_inv), np.ones((dim,)))
    B = np.dot(np.dot(mu, Sigma_inv), np.ones((dim,)))
    C = np.dot(np.dot(mu, Sigma_inv), mu)
    t_w = np.dot(Sigma_inv, mu - rf)/(B - A*rf)
    t_m = (C - B*rf)/(B - A*rf)
    t_sigma = np.sqrt((C - 2*B*rf + A*(rf**2))/((B - A*rf)**2))
    t_slope = (t_m - rf)/t_sigma
    return t_w, t_sigma, t_slope

class Var:
    
    from scipy.stats import norm
    
    def __init__(self, weight, mu, sigma, alpha, corr):
        self.weight = weight
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.corr = corr
        
    def Sigma(self):
        S = np.diag(self.sigma)
        big_sigma = np.dot(np.dot(S, self.corr), S)
        return big_sigma
    
    def var(self):
        big_sigma = self.Sigma()
        var = -np.dot(self.weight, self.mu) + \
              norm(0,1).ppf(self.alpha)*np.sqrt(np.dot(np.dot(self.weight, big_sigma), self.weight.T))
        return float(var)
    
    def es(self):
        big_sigma = self.Sigma()
        c = np.exp((-1.0/2)*(norm(0,1).ppf(self.alpha)**2))/((1-self.alpha)*np.sqrt(2*np.pi))
        es = c*np.sqrt(np.dot(np.dot(self.weight, big_sigma), self.weight.T)) - np.dot(self.weight, self.mu)
        return float(es)
    
if __name__ == '__main__':
    
    mu = np.array([0.04, 0.08, 0.12, 0.15])
    sigma = np.array([0.07, 0.12, 0.18, 0.26])
    corr = np.array([[1,0.2,0.5,0.3], [0.2,1,0.7,0.4],[0.5,0.7,1,0.9],[0.3,0.4,0.9,1]])
    rf = np.arange(0.01,0.045, 0.005)
    alpha = 0.99
    
    for each in rf:
        
        t_w, t_sigma, t_slope = tangency_portfolio_opt(mu, sigma, corr, each)
        portfolio_risk = Var(t_w, mu, sigma, alpha, corr)
        
        print 'rf equals to ',each
        print 'weight', t_w
        print 'sigma', t_sigma
        print 'slope', t_slope
        print 'the portfolio var is ',portfolio_risk.var()
        print 'the portfolio es is', portfolio_risk.es()
        print ""
