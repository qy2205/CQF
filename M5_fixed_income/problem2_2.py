# -*- coding: utf-8 -*-
"""
# CQF Module 5 Exam Problem 2 (2)
@author: QUAN YUAN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prob_survival(D, RR, dt, S, N):
    '''
    Input:
    D: discount factor
    RR: Recover Rate
    dt: delta t
    S: credit spread
    N: N-period
    
    Output:
    P(T1)——P(TN) list type
    '''
    P = np.zeros(N+1)
    L = 1 - RR
    if N == 0:
        P[0] = 1
    elif N == 1:
        P[0] = 1
        P[N] = L*1.0/(L + dt*S[0])
    else:
        P[0] = 1
        P[1] = L*1.0/(L + dt*S[0])
        count = 2
        
        while count <= N:
            # sigma part
            sigma = 0
            for i in range(1,count):
                sigma += D[i]*(L*P[i-1] - (L+dt*S[count-1])*P[i])
            P[count] = sigma*1.0/(D[count]*(L + dt*S[count-1])) + \
                       P[count-1]*L/(L + dt*S[count-1])
            count += 1
    return P

if __name__ == '__main__':
    # Question 2
    n = 5
    RR = 0.4
    dt = 1
    S = np.array([141.76, 165.36, 188.56, 207.32, 218.38])/10000
    T = [0, 1, 2, 3, 4, 5]
    rf = 0.8/100
    DF = []
    for i in range(n+1):
        DF.append(np.exp(-rf*i))
    df = pd.DataFrame({'T':T, 'DF': DF, 'survival_prob': prob_survival(DF, RR, dt, S, n)})
    print "Question 2"
    print "Survival probabilities for DB bank: "
    print df
    
    # Question 3
    P = df['survival_prob']
    hazard = np.zeros(len(P))
    for i in range(1, len(P)):
        hazard[i] = (-1.0/dt)*(np.log(P[i]) - np.log(P[i-1]))
    print "Hazard Rate Term Structure"
    print hazard
    # visualization
    plt.figure(figsize = (10, 6))
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Lambda', fontsize = 16)
    plt.ylim(0, 0.05)
    plt.title('Hazard Rate Term Structure', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.bar(range(1, len(hazard)), hazard[1:], label = 'hazard rate')
    plt.show()
    
    # pdf
    t = np.arange(1, 5.01, 0.01)
    pdf = []
    for each_t in t:
        floor_t = np.floor(each_t)
        each_lambda = hazard[int(floor_t)]
        pdf.append(each_lambda*np.exp(-each_lambda*each_t))
    
    # visualization
    plt.figure(figsize = (10, 6))
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Lambda', fontsize = 16)
    plt.ylim(0, 0.05)
    plt.title('Exponential pdf of lambda $f(t) = \lambda e^{-\lambda t}$', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.plot(t, pdf, label = 'pdf')
    plt.show()
