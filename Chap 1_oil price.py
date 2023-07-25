#%%
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# data source:
# U.S. Energy Information Administration, Crude Oil Prices: Brent — Europe [DCOILBRENTEU]
# retrieved from FRED, Federal Reserve Bank of St. Louis;
# https://fred.stlouisfed.org/series/DCOILBRENTEU, June 10, 2022.

start = dt.datetime(2018, 7, 1)
end = dt.datetime(2023, 7, 1)
# end = dt.date.today()
brent_oil_price = web.DataReader('DCOILBRENTEU', 'fred', start, end)

#%%
figure = plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(1,1,1)
ax_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
brent_oil_price = brent_oil_price.dropna()
plt.figure(figsize=(10, 6))
plt.plot(brent_oil_price)
# plt.grid(axis="y")
plt.xlabel("Date")
plt.ylabel("Oil Price ($/Barrel)")
plt.show()

