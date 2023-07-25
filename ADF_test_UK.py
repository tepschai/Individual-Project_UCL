'''
https://blog.csdn.net/FrankieHello/article/details/86766625

The null hypothesis that a unit root is present in a time series sample, 
in other words, the time series sample is not stable (instead, a random walk)
    
    Returns
    -------
    adf : float
        Test statistic
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010)
    usedlag : int
        Number of lags used
    nobs : int
        Number of observations used for the ADF regression and calculation of
        the critical values
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010)
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes

'''
#%%
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(data):
    result = adfuller(data)
    # result[1] is the p-value, the smallerb the more likely to reject the null hypothesis
    print(f'The p-value equals to: {result[1]}')
    if result[1] < 0.05:
        print('The data is stable.')
    else:
        print('The data is unstable.')
    print(result)
    print('\n')

#%%
if __name__ == '__main__':
    
    cpi = pd.read_csv(r'..\Data process\1989_2023_cpi.csv')
    unenmpoyment = pd.read_csv(r'..\Data process\Unemployment rate (aged 16 and over, seasonally adjusted).csv')
    
    print('ADF test of CPI:')
    adf_test(cpi['CPI'].values)
    print('ADF test of Unemployment:')
    adf_test(unenmpoyment['Unemployment'].values)
    
    


