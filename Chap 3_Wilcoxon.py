
#%%
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

#%%
def wilcoxon_signed_rank_test(x, y, n):
    stat, p = wilcoxon(x.values[n:], y.values[n:])
    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    # alpha = 0.05
    # #H0：Same distribution
    # if p > alpha:
    #     print(f'Same distribution (fail to reject H0) of {x.name} and {y.name}')
    # else:
    #     print(f'Different distribution (reject H0) of {x.name} and {y.name}')
    return stat, p

#%%
def data_analysis(data, n, i, j, records_raw, records_stat, records_jud):
    stat, p = wilcoxon_signed_rank_test(data[i], data[j], n)
    records_raw[i][j] = p
    records_stat[i][j] = stat
    records_jud[i][j] = 'Pass' if p > 0.05 else 'Rejected'
    

#%%

if __name__ == '__main__':
    
    data_raw_cpi = pd.read_csv('cpi_rank.csv')
    data_cpi = data_raw_cpi.dropna(how='any').reset_index(drop=True)
    
    data_raw_un = pd.read_csv('un_rank.csv')
    data_un = data_raw_un.dropna(how='any').reset_index(drop=True)
    
    index = list(['1', '2', '3', '-1', '-2', '-3', 'SMA', 'Phillips'])
    records_raw_cpi = pd.DataFrame(np.full([8,8], np.nan), index=index, columns=index)
    records_stat_cpi = pd.DataFrame(np.full([8,8], np.nan), index=index, columns=index)
    records_jud_cpi = pd.DataFrame(np.full([8,8], np.nan), index=index, columns=index)
    
    records_raw_un = pd.DataFrame(np.full([8,8], np.nan), index=index, columns=index)
    records_stat_un = pd.DataFrame(np.full([8,8], np.nan), index=index, columns=index)
    records_jud_un = pd.DataFrame(np.full([8,8], np.nan), index=index, columns=index)
    
    n = -24

    for i in index:
        for j in index:
            if i == j: 
                pass
            else:
                data_analysis(data_cpi, n, i, j, records_raw_cpi, records_stat_cpi, records_jud_cpi)
                data_analysis(data_un, n, i, j, records_raw_un, records_stat_un, records_jud_un)
    
    records_raw_cpi.to_csv(f'cpi_wilcoxion_{n}.csv')
    records_raw_un.to_csv(f'un_wilcoxion_{n}.csv')
                
    print(records_raw_cpi)
    print(records_jud_cpi)
    print('\n')
    print(records_raw_un)
    print(records_jud_un)
