import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import datetime as datetime
import matplotlib.dates as mdates
import seaborn; seaborn.set()
import numpy.fft as ft

from scipy.optimize import curve_fit

df = pd.read_csv('NFLX.csv')
df.head()

df.info()

df['Date'] = pd.to_datetime(df['Date'])

df.info()

dfc = df.copy()
dfc.set_index('Date', inplace=True)

dfM = dfc.resample('M').mean()
dfW = dfc.resample('W-MON').mean()

dfW.head()

dfM.head()

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(df['Date'],df['Close'], c='blue', label='Close Price')

plt.legend(loc='upper left')
plt.title('Netflix Stock')
plt.xlabel('Year')
plt.ylabel('Price (USD)')

plt.show()

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(dfW['Close'], label='Weekly Close Price')
plt.plot(dfM['Close'], label='Monthly Close Price')

plt.legend(loc='upper left')
plt.title('Netflix Stock - Means')
plt.xlabel('Year')
plt.ylabel('Price (USD)')

plt.show()

N = len(df['Date'])

c_values = df['Close'].values
d_valuess = df['Date'].values
d_values = []
l_return = []

for i in range(N-1):
  l_return.append(c_values[i + 1]/c_values[i] - 1)
  d_values.append(d_valuess[i])

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values,l_return, c='midnightblue', label='Linear Return')

plt.legend(loc='upper left')
plt.title('Daily Linear Return ($R_t$) of Netflix Stock')
plt.xlabel('Year')
plt.ylabel('Percent')

plt.show()

log_return = []

for i in range(N-1):
  log_return.append(np.log(c_values[i+1]) - np.log(c_values[i]))

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, log_return, c='navy', label='Log Return')

plt.legend(loc='upper left')
plt.title('Daily Log Return ($r_t$) of Netflix Stock')
plt.xlabel('Year')
plt.ylabel('Percent')

plt.show()

diff = []

for i in range(N-1):
  diff.append(log_return[i]-l_return[i])

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values, diff, c='darkblue', label='Difference Between Returns')

plt.legend(loc='upper left')
plt.title('Difference Between Daily Linear Return and Log Return of Netflix Stock')
plt.xlabel('Year')
plt.ylabel('Percent')

plt.plot()


fig, ax = plt.subplots(figsize=(16, 9))

x = np.arange(0, 10, 0.1)
y = np.log(1+x)
z = 1*x

ax.plot(y, color='red', label='$f(x) = log(1+x)$')
ax.plot(z, color='black', label='$f(x) = x$')

plt.legend(loc='best')
plt.title('Linear and Logarithm Function')
plt.xlabel('X')
plt.ylabel('f(x)')


plt.show()

n_return = []

mean = np.average(log_return)
stdev = np.std(log_return)

for i in range(N-1):
  n_return.append((log_return[i]-mean)/stdev)

fig, ax = plt.subplots(figsize=(16, 9))

plt.plot(d_values,n_return, c='royalblue', label='N Return')

plt.legend(loc='upper left')
plt.title('Daily Normalized Return of Netflix Stock')
plt.xlabel('Year')
plt.ylabel('Percent')

plt.plot()

fig, ax = plt.subplots(figsize=(16, 9))
plt.hist(log_return, bins=100, density=True)

plt.show()

def normal_dist(x, mu, msd, A):
    return (A/msd*np.sqrt(2*np.pi))*(np.exp(-0.5*((x-mu)/msd)**2))

fig, ax = plt.subplots(figsize=(16, 9))

count, bins, ignored = plt.hist(log_return, bins=100, density=True)

bins = np.array(bins)
bins = bins[:-1]

fit_values, co_matrix = curve_fit(normal_dist, bins, count) 

L, M, N = fit_values
plt.plot(bins, normal_dist(bins, L, M, N), '--r')
plt.show()


r_squared_l = []

for i in range(len(count)):
    r_squared_l.append((normal_dist(bins[i],L,M,N) - count[i])**2)

R_squared = sum(r_squared_l)/len(r_squared_l) - 1 

print(R_squared)

fig, ax = plt.subplots(figsize=(16, 9))

f = ft.fft(log_return)
s = f*np.conj(f)
c = ft.ifft(s).real

plt.plot(d_values,c)
plt.show()

