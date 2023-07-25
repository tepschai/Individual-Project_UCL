'''
Data process for individual project
ver_1: 24th May 2023
Chai, Lei
'''
#%%
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.optimize import curve_fit

#%%
#CPI and unemployment

cpi = pd.read_csv('1989_2023_cpi.csv')
cpi['Date'] = pd.to_datetime(cpi['Date'])
cpi.set_index('Date', inplace=True)
# cpi = cpi.apply(lambda x:x.astype(float))

unemploy = pd.read_csv('Unemployment rate (aged 16 and over, seasonally adjusted).csv')
unemploy['Date'] = pd.to_datetime(unemploy['Date'])
unemploy.set_index('Date', inplace=True)
# unemploy = unemploy.apply(lambda x:x.astype(float))

#%%
#CPI figure

#SMA, std
sma_cpi_12 = cpi.rolling(12).mean()
sma_cpi_6 = cpi.rolling(6).mean()
std_cpi_12 = cpi.rolling(12).std()
std_cpi_6 = cpi.rolling(6).std()

figure_1 = plt.figure(figsize = (10, 5))
ax_1 = plt.subplot(1,1,1)
plt.plot(cpi, label='CPI in the UK')
plt.plot(sma_cpi_12, label='12-month SMA', linestyle='-.')
plt.plot(sma_cpi_6, label='6-month SMA', linestyle='--')
plt.plot(std_cpi_12, label='12-month moving Std')
plt.plot(std_cpi_6, label='6-month moving Std', linestyle = '-.')
plt.ylabel('%')
# plt.xlabel('Time')
ax_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(cpi.index[0::40])
plt.legend(loc='best')
plt.savefig('Inflation.png',bbox_inches='tight')
plt.show()

#%%
# Unemployment figure

#SMA, std
sma_unemploy_12 = unemploy.rolling(12).mean()
sma_unemploy_6 = unemploy.rolling(6).mean()
std_unemploy_12 = unemploy.rolling(12).std()
std_unemploy_6 = unemploy.rolling(6).std()

figure_2 = plt.figure(figsize = (10, 5))
ax_1 = plt.subplot(1,1,1)
plt.plot(unemploy, label='Unemployment in the UK')
plt.plot(sma_unemploy_12, label='12-month SMA', linestyle='-.')
plt.plot(sma_unemploy_6, label='6-month SMA', linestyle='--')
plt.plot(std_unemploy_12, label='12-month moving Std')
plt.plot(std_unemploy_6, label='6-month moving Std', linestyle = '-.')
plt.ylabel('%')
# plt.xlabel('Time')
ax_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(unemploy.index[0::40])
plt.legend(loc='best')
plt.savefig('Unemployment.png',bbox_inches='tight')
plt.show()

#%%
#CPI VS UNEMPLOYMENT
method = 'inner'
cpi_unem = pd.merge(cpi, unemploy, how=method, on='Date')
cpi_unem.sort_index(inplace=True)

#correlation calculation
correlation = cpi_unem.corr()
print(correlation)
print('\n')

corr_roll_12 = cpi_unem.rolling(12).corr()
corr_12 = corr_roll_12['CPI'].loc[:,'Unemployment']

print(f'Number of positive correlation: {len(corr_12.loc[corr_12>=0])}')
print(f'Number of negative correlation: {len(corr_12.loc[corr_12<0])}')
print(f'Percentage of positive correlation: {len(corr_12.loc[corr_12>=0])/len(corr_12.loc[corr_12.notnull()])}')
#
# #%%
# #correlaiton
# figure_2 = plt.figure(figsize = (10, 5))
# ax_2 = plt.subplot(1,1,1)
# plt.plot(cpi_unem['CPI'], label='CPI in the UK')
# plt.plot(cpi_unem['Unemployment'], label='Unemployment in the UK')
# plt.plot(corr_12, label='12-month correlation',linestyle='--')
# plt.hlines(0, cpi_unem.index[0], cpi_unem.index[-1], linestyles='dotted', color='red')
# plt.ylabel('%')
# # plt.xlabel('Date')
# plt.xticks(cpi.index[0::40])
# ax_2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.legend(loc='best')
# plt.savefig('CPI_unemployemnt.png', bbox_inches='tight')
# plt.show()

#%%
# hist of correlation
c_v = corr_12.loc[corr_12.notnull()].values
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(111)
plt.hist(x=c_v, bins = 30, edgecolor='w', label = 'Frequency')
ax1.set_ylabel('Frequency')
plt.legend(loc=2)

ax2 = ax1.twinx()
ax1.set_ylabel('KDE')
sns.kdeplot(c_v, fill=True, label='KDE')
plt.legend(loc=1)
plt.savefig('correlation_distribution.png', bbox_inches='tight')
plt.show()

#%%
# cloud and fit
def func(x, a ,b, c):
    return b*(x**c) - a

xdata_60 = cpi_unem['Unemployment'][-60:].values
ydata_60 = cpi_unem['CPI'][-60:].values
popt_60, pcov_60 = curve_fit(func, xdata_60, ydata_60)
ypredict_60 = func(xdata_60, *popt_60)

figure_3 = plt.figure(figsize = (20, 6))
plt.subplot(1,3,1)
plt.title('(a)', y=-0.15)
plt.scatter(cpi_unem['Unemployment'], cpi_unem['CPI'])
plt.xlabel('Unemployment')
plt.ylabel('CPI')

plt.subplot(1,3,2)
plt.title('(b)', y=-0.15)
plt.scatter(cpi_unem['Unemployment'][-120:], cpi_unem['CPI'][-120:])
plt.xlabel('Unemployment')
plt.ylabel('CPI')

plt.subplot(1,3,3)
plt.title('(c)', y=-0.15)
plt.scatter(cpi_unem['Unemployment'][-60:], cpi_unem['CPI'][-60:])
plt.plot(xdata_60, ypredict_60, 'r--', label='Fit curve')
plt.xlabel('Unemployment')
plt.ylabel('CPI')

plt.legend(loc='best')
plt.savefig('scatter.png', bbox_inches='tight')
plt.show()

print('fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_60))

#%%
# evaluate the fit
def getIndice(y_predict, y_data):
    n = y_data.size
    SSE=((y_data-y_predict)**2).sum()
    MSE=SSE/n
    RMSE=np.sqrt(MSE)

    u = y_data.mean()
    SST=((y_data-u)**2).sum()
    SSR=SST-SSE
    R_square=SSR/SST
    return SSE, MSE, RMSE, R_square

indice = getIndice(ypredict_60, ydata_60)
print(f'SSE：{indice[0]}, MSE: {indice[1]}, RMSE: {indice[2]}, R^2: {indice[3]}')

