#!pip install arch
#!pip install statsmodels

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import datetime as datetime
import matplotlib.dates as mdates
import seaborn; seaborn.set()
from arch import arch_model
import statsmodels as sm

df = pd.read_csv('NFLX.csv')
df.head()


df.info()





df['Date'] = pd.to_datetime(df['Date'])


df.info()


## Returns ##


##preps
N = len(df['Date'])

c_values = df['Close'].values
d_valuess = df['Date'].values
d_values = []
l_return = []
log_return = []
n_return = []

##linear return
for i in range(N-1):
  l_return.append(c_values[i + 1]/c_values[i] - 1)
  d_values.append(d_valuess[i])

##log return
for i in range(N-1):
  log_return.append(np.log(c_values[i+1]) - np.log(c_values[i]))

##normalized return
mean = np.average(log_return)
stdev = np.std(log_return)

for i in range(N-1):
  n_return.append((log_return[i]-mean)/stdev)

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, l_return, c='green', label='Linear Return')

plt.legend(loc='upper right')
plt.title('Daily Returns of Netflix Stock', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Return')

plt.show()

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, n_return, label='Normalized Return')

plt.legend(loc='upper right')
plt.title('Daily Returns of Netflix Stock', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Return')

plt.show()


fig, ax = plt.subplots(figsize=(16, 9))
lags = 100

ACF = plt.acorr(l_return, maxlags=lags, label="Autocorrelation") 

plt.title('Autocorrelation of Linear Return', fontsize=16)
plt.xlabel('Lags (k)')
plt.ylabel('$r_{xx}$ (k)')
plt.legend(loc="best")

plt.show()


R = []
phi = []
zero = int((len(ACF[1])-1)/2)
line = 0

while (line<zero):
    row = []
    for j in range(zero):
      row.append(ACF[1][zero + j - line])
    R.append(row)
    line = line + 1

rxx = ACF[1][(zero+1):(zero+1+lags)]
R_inverse = np.linalg.inv(np.array(R)) 

phi = np.matmul(R_inverse, rxx)
phi = np.insert(phi, 0, 1)


fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(phi, label="PACF", marker='o', color='red')
plt.title('Partial Autocorrelation Function', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Lags (k)')
plt.ylabel('PACF')

plt.show()


mu = 0 
sigma = 0.1
epsilon = np.random.normal(mu, sigma, N)

y_ar = []

for i in range(N-1):
  sum = 0
  n = 1
  for n in range(1, len(phi)-1):
    sum = phi[n]*n_return[i-n] + epsilon[n] + sum
  y_ar.append(sum)


fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, n_return, label="Normalized Return")
plt.plot(d_values, y_ar, label="AR Model", alpha=0.9)

plt.title('Autoregressive Model', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Return')

plt.show()


var_xt  = np.var(n_return)
var_e = np.var(epsilon)

theta = ((var_xt/var_e) - 1 )**(1/2)
arma = []

for j in range(len(y_ar)):
  arma.append(y_ar[j] + theta*epsilon[j])


fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, n_return, label="Normalized Return")
plt.plot(d_values, arma, label="ARMA with $Q = 1$", alpha=.9)

plt.title('ARMA Model', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Return')

plt.show()


r_total = 0
residue = []

for i in range(len(arma)):
  r = (n_return[i] - arma[i])**2
  r_total = r + r_total
  residue.append(r)

print(np.sqrt(r_total))


fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, residue, label="Residue")

plt.title('Residue', fontsize=16)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Residue')

plt.show()

## GARCH Model ##

garch_model = arch_model(log_return, vol='Garch', p=1, q=1, dist='Normal')
model_fit = garch_model.fit()
estimation = model_fit.forecast()

print(model_fit)


omega = 7.0346e-05
alpha = 0.05
beta  = 0.85


sig_garch = []

for i in range(len(n_return)):
  sig_garch.append(omega + alpha*(l_return[i-1]**2) + beta*(varGARCH[i-1]**2))


fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, sig_garch, color="royalblue", label="Volatility")

plt.title('Volatility of Log Return by GARCH Model',fontsize=16)
plt.ylabel('Volatility')
plt.xlabel('Date')
plt.legend()

plt.show()


## Markov Chain ##


mc = pd.DataFrame()
mc['labels'] = pd.qcut(n_return, 3, labels=["L", "M", "H"])
mc = mc.sort_index()

col = []

for i in range(0, len(markov_df)): 
    if markov_df['labels'][i]=='L':
        col.append('green')   
    if markov_df['labels'][i]=='M': 
        col.append('red') 
    if markov_df['labels'][i]=='H':
        col.append('blue')  

fig, ax = plt.subplots(figsize=(16, 9))
plt.title('High, Mean and Low Market', fontsize=16)

for i in range(len(markov_df)):
    plt.scatter(x=markov_df.index[i], y=markov_df['labels'][i], s=50, c=col[i])

plt.plot(markov_df['labels'], color="black", alpha=0.2)
plt.yticks()

plt.show()


def build_transition_grid(labelSeries, label):
    list_seq = []
    for i in range(len(labelSeries)-1):
            list_seq.append(labelSeries[i] + '-' + labelSeries[i+1])
    
    nlabels = pd.Series(data=list_seq)
    uniqueValues = nlabels.unique()
    count = nlabels.value_counts(normalize=True)
    print(count)

build_transition_grid(markov_df['labels'], ['L', 'H', 'M'])


