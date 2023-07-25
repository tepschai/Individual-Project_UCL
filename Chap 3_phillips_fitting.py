#%%
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

cpi = pd.read_csv('1989_2023_cpi.csv')
cpi['Date'] = pd.to_datetime(cpi['Date'])
cpi.set_index('Date', inplace=True)

unemploy = pd.read_csv('Unemployment rate (aged 16 and over, seasonally adjusted).csv')
unemploy['Date'] = pd.to_datetime(unemploy['Date'])
unemploy.set_index('Date', inplace=True)

method = 'inner'
cpi_unem = pd.merge(cpi, unemploy, how=method, on='Date')
cpi_unem.sort_index(inplace=True)

#%%
# fit
def func(x, a ,b, c):
    return b*(x**c) - a

un_data_60 = cpi_unem['Unemployment'][-60:].values
cpi_data_60 = cpi_unem['CPI'][-60:].values

popt_60, pcov_60 = curve_fit(func, un_data_60, cpi_data_60)

cpi_pred_60 = func(un_data_60, *popt_60)
un_pred_60 = ((cpi_data_60 + popt_60[0])/popt_60[1])**(1/popt_60[2])

#%%
results = cpi_unem[-60:].copy()
results['CPI_pred'] = cpi_pred_60
results['Un_pred'] = un_pred_60
results['Un_pred'].fillna(method='ffill', inplace=True)

#%%
def getIndice(y_predict, y_data):
    n = y_data.size
    SSE=((y_data-y_predict)**2).sum()
    MSE=SSE/n
    RMSE=np.sqrt(MSE)
    return RMSE

#%%
print('RMSE of CPI:')
print(getIndice(results['CPI'], results['CPI_pred']))
print('\n')
print('RMSE of Unemployment:')
print(getIndice(results['Unemployment'], results['Un_pred']))

#%%

