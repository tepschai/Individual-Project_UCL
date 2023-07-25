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
import seaborn as sns

#%%
# CPI
cpi = pd.read_csv('1989_2023_cpi.csv')
cpi['Date'] = pd.to_datetime(cpi['Date'])
cpi.set_index('Date', inplace=True)
cpi = cpi.apply(lambda x:x.astype(float))

#%%
# unemployment
unemploy = pd.read_csv('Unemployment rate (aged 16 and over, seasonally adjusted).csv')
unemploy['Date'] = pd.to_datetime(unemploy['Date'])
unemploy.set_index('Date', inplace=True)
unemploy = unemploy.apply(lambda x:x.astype(float))


#%%
# Oil and other fuels
fuels = pd.read_csv('Oil and other fuels.csv')
fuels['Date'] = pd.to_datetime(fuels['Date'])
fuels.set_index('Date', inplace=True)

#%%
# Average Weekly Earnings (AWE)
awe = pd.read_csv('AWE_0613.csv')
awe['Date'] = pd.to_datetime(awe['Date'])
awe.set_index('Date', inplace=True)

#%%
# Bank rate
bank_oringin = pd.read_csv('Bank of England Database.csv')
bank = bank_oringin.iloc[:,:2]
bank['Date'] = pd.to_datetime(bank['Date'])
bank['Date'] = bank['Date'] + dt.timedelta(days=1)
bank['Date'] = bank['Date'].shift(-1)
bank.set_index('Date', inplace=True)
bank.sort_index(inplace=True)
bank.columns = ['Bank rate']

#%%
method = 'inner'
cpi_unem = pd.merge(cpi, unemploy, how=method, on='Date')
cpi_unem.sort_index(inplace=True)

#%%
# combine all the data together
method = 'inner'
data_1 = pd.merge(cpi, unemploy, how=method, on='Date')
data_2 = pd.merge(data_1, fuels, how=method, on='Date' )
data_3 = pd.merge(data_2, awe, how=method, on='Date' )
data_4 = pd.merge(data_3, bank, how=method, on='Date' )
data_4.sort_index(inplace=True)

# #%% correlation matrix
# corr_all = data_4.corr()
# corr_five = data_4.iloc[-60:,:].corr()
# corr_three = data_4.iloc[-36:,:].corr()
#
# fig_cor_1 = plt.figure(figsize=(8,7))
# # plt.subplot(1,3,1)
# sns.heatmap(corr_all, annot=True, cmap='RdBu', xticklabels=1, yticklabels=1)
# plt.savefig('correlation_all.png',bbox_inches='tight')
# # plt.title('(a)')
#
# fig_cor_2 = plt.figure(figsize=(8,7))
# # plt.subplot(1,3,2)
# sns.heatmap(corr_five, annot=True, cmap='RdBu', xticklabels=1, yticklabels=1)
# plt.savefig('correlation_5.png',bbox_inches='tight')
# # plt.title('(b)')
#
# fig_cor_3 = plt.figure(figsize=(8,7))
# # plt.subplot(1,3,3)
# sns.heatmap(corr_three, annot=True, cmap='RdBu', xticklabels=1, yticklabels=1)
# plt.savefig('correlation_3.png',bbox_inches='tight')
# # plt.title('(c)')
#
# # plt.savefig('correlation_5.png',bbox_inches='tight')
# plt.show()

#%%
data_4.to_csv('../training_data.csv')