
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import datetime as datetime
import matplotlib.dates as mdates
import seaborn; seaborn.set()
import statsmodels as sm

amzn = pd.read_csv('AMZN.csv')
dis  = pd.read_csv('DIS.csv')
intc = pd.read_csv('INTC.csv')
nflx = pd.read_csv('NFLX.csv')
nsdq = pd.read_csv('NSDQ.csv')
tsla = pd.read_csv('TSLA.csv')

amzn['Date'] = pd.to_datetime(amzn['Date'])
dis['Date']  = pd.to_datetime(dis['Date'])
intc['Date'] = pd.to_datetime(intc['Date'])
nflx['Date'] = pd.to_datetime(nflx['Date'])
nsdq['Date'] = pd.to_datetime(nsdq['Date'])
tsla['Date'] = pd.to_datetime(tsla['Date'])


def normalized_return(datafile):
    close_values = datafile['Close'].values
    data_e_values  = datafile['Date'].values
    N = len(close_values)

    data_values = []
    log_return = []
    n_return = []

    ##log return
    for i in range(N-1):
        log_return.append(np.log(close_values[i+1]) - np.log(close_values[i]))
        data_values.append(data_e_values[i])

    ##normalized return
    mean = np.average(log_return)
    stdev = np.std(log_return)

    for i in range(N-1):
        n_return.append((log_return[i]-mean)/stdev)    

    ##return
    return np.array(data_values), np.array(n_return)



amzn_d, amzn_r = normalized_return(amzn)
dis_d,  dis_r  = normalized_return(dis)
intc_d, intc_r = normalized_return(intc)
nflx_d, nflx_r = normalized_return(nflx)
nsdq_d, nsdq_r = normalized_return(nsdq)
tsla_d, tsla_r = normalized_return(tsla)



fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(amzn_d, amzn_r, label='Amazon')
plt.plot(dis_d, dis_r, label='Disney')
plt.plot(intc_d, intc_r, label='Intel')
plt.plot(nflx_d, nflx_r, label='Netflix')
plt.plot(tsla_d, tsla_r, label='Tesla')

plt.legend(loc='upper right')
plt.title('Daily Normalized Returns', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Return')

plt.show()


def beta(df1, df2):
    return [[np.cov(df1, df2)[0][1]/np.var(df2)],
            [np.mean(df1)]]

amzn_c = beta(amzn_r, nsdq_r)
dis_c  = beta(dis_r, nsdq_r)
intc_c = beta(intc_r, nsdq_r)
nflx_c = beta(nflx_r, nsdq_r)
tsla_c = beta(tsla_r, nsdq_r)

r_0 = 0.0
rm_nflx = np.mean(nsdq_r)



fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(amzn_c[0], amzn_c[1], 'o', label='Amazon')
plt.plot(dis_c[0],  dis_c[1],  'o', label='Disney')
plt.plot(intc_c[0], intc_c[1], 'o', label='Intel')
plt.plot(nflx_c[0], nflx_c[1], 'o', label='Netflix')
plt.plot(tsla_c[0], tsla_c[1], 'o', label='Tesla')
plt.plot(np.arange(0.15,0.75,0.001),
         [r_0 + i*(rm_nflx-r_0) for i in np.arange(0.15,0.75,0.001)],
         label="SML")

plt.title('CAPM Model', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Risk')
plt.ylabel('Return')

plt.show()



alpha = np.array([[amzn_r.mean()], [nflx_r.mean()]])
r = np.array([[amzn_r], [nflx_r]])

S = np.cov([amzn_r, nflx_r])
S_i = np.linalg.inv(S)

e  = np.array([[1], [1]])
e_t = np.transpose(e)

M = np.array([[e.T.dot(S_i.dot(e))[0][0], alpha.T.dot(S_i.dot(e))[0][0]],
            [e.T.dot(S_i.dot(alpha))[0][0], alpha.T.dot(S_i.dot(alpha))[0][0]]])
M_i = np.linalg.inv(M)



r_e = [] 
risk=[]
portfolio=[0,0,0]

for i in np.arange(-0.0001,0.001,0.0000001):
    r_e.append(i)
    risk_i=np.sqrt(np.array([1, i]).dot(M_i.dot(np.array([[1], [i]])))[0])
    risk.append(risk_i)
    ratio = i/risk_i

    if ratio > portfolio[0]:
        portfolio[0] = ratio
        portfolio[1] = i 
        portfolio[2] = risk_i

alpha_zero_mu = M_i.dot(np.array([[1],[portfolio[1]]]))
w_0 = alpha_zero_mu[0]*(S_i.dot(e) + alpha_zero_mu[1]*(S_i.dot(alpha)))

sum = 0
aux = []

for i in w_0:
    sum += i

for i in w_0:
    aux.append(max(0, i/sum))

w_0 = aux



fig, ax = plt.subplots(figsize=(16, 9))

#plt.plot(np.std(amzn['Close']), np.mean(amzn['Close']), 'o', label='Amazon')
#plt.plot(np.std(nflx['Close']), np.mean(nflx['Close']), 'o', label='Netflix')

plt.plot(risk, r_e, label='Markowitz Bullet')
plt.plot(portfolio[2], portfolio[1], 'o', label='Best Portfolio')

plt.legend(loc='upper right')
plt.title('Markowitz', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Risk')
plt.ylabel('Return')

plt.show()

