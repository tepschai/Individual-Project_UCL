# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:32:07 2023

@author: chai_l
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

#%%

def data_processing(data, obj, lag):
    
    dim = data.shape[1]
    step = 1

    scaler = MinMaxScaler(feature_range=(0, 1))

    def datareshape(data, obj, lag, step, dim):

        values = data.values
        forecast = data[obj]
        # print(values.shape)

        num = lag*dim
        x = np.zeros((values.shape[0]-lag-step+1,num))
        y = np.zeros((values.shape[0]-lag-step+1,step))

        for i in range(0,values.shape[0]-lag-step+1):
            x[i] = np.array(values[i:i+lag]).reshape([1,num])
            y[i] = np.array(np.array(forecast)[i+lag:i+lag+step]).reshape(1,step)

        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y)

        x = x.reshape(x.shape[0],x.shape[1],1)

        print(x.shape)
        print(y.shape)

        return x, y

    x, y = datareshape(data, obj, lag, step, dim)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, shuffle=True)
    
    return scaler, x, y, x_train, x_test, y_train, y_test
    
#%%
def lstm(scaler, x, y, x_train, x_test, y_train, y_test, obj, optimizer, optimizer_name, lag, net):

    batch_size = 60
    # net = 64
    epochs = 500

    model = Sequential()
    model.add(LSTM(units=net, activation='relu', return_sequences=True, input_shape=(x.shape[1],x.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units=net, activation='relu', return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(units=net/2))
    model.add(Dense(units=1))
    print(model.summary())

    # optimizer = 'adam'

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    callback = EarlyStopping(monitor='val_loss',patience=15, restore_best_weights=True)
    History = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2,
                        callbacks=[callback])

    model.fit(x_train, y_train, validation_split=0.2, callbacks=[callback], epochs=epochs)

    fig_train = plt.figure()
    plt.plot(History.history['loss'], label='loss')
    plt.plot(History.history['val_loss'], label = 'val loss')
    plt.legend(loc='best')
    plt.savefig(f'{obj}_{optimizer_name}_{lag}_{net}_loss.png')
    plt.show()

    y_pred_norm = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred_norm)
    y_test = scaler.inverse_transform(y_test)

    fig_contrast = plt.figure()
    plt.plot(y_test, label='test', marker='o')
    plt.plot(y_pred, label='predition', marker='o')
    plt.ylabel('%')
    plt.legend(loc='best')
    plt.savefig(f'{obj}_{optimizer_name}_{lag}_{net}_test.png')
    plt.show()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')

    # data['Date'] = pd.to_datetime(data['Date'])
    # data.set_index('Date', inplace=True)
    y_true = data[obj]
    xx_test = x[-round(x.shape[0]*0.2):]
    yy_pred_norm = model.predict(xx_test)
    yy_pred = scaler.inverse_transform(yy_pred_norm)
    yy_pred_dt = pd.DataFrame(yy_pred, index=y_true.index[-round(x.shape[0]*0.2):])

    mse_pred = mean_squared_error(y_true.values[-round(x.shape[0]*0.2):], yy_pred)
    rmse_pred = np.sqrt(mse_pred)
    print(f'MSE_pred: {mse_pred}')
    print(f'RMSE_pred: {rmse_pred}')
    
    # record results
    record = pd.DataFrame({'Optimizer':[optimizer_name], 'Lag size':[lag], 'Neurons':[net], 'MSE':[mse], 
              'RMSE':[rmse], 'MSE_pred':[mse_pred], 'RMSE_pred':[rmse_pred]})
    
    # draw figures    
    fig_pre = plt.figure(figsize=(10, 5))
    ax_pre = plt.subplot(1,1,1)
    ax_pre.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_pre.set_xticks(y_true.index[::40])
    plt.plot(y_true, label='True value')
    plt.plot(yy_pred_dt, label='Prediction', marker='v', alpha=0.5)
    plt.legend(loc='best')
    plt.ylabel('%')
    plt.savefig(f'{obj}_{optimizer_name}_{lag}_{net}_pred.png')
    plt.show()

    # save raw data
    test = pd.DataFrame({'y_test':y_test.reshape(y_test.shape[0],), 'y_pred':y_pred.reshape(y_pred.shape[0],)})
    test.to_csv(f'{obj}_{optimizer_name}_{lag}_{net}_test.csv')
    pred = pd.DataFrame({'y_true':y_true.values[-round(x.shape[0]*0.2):], 'yy_pred':yy_pred.reshape(yy_pred.shape[0],)})
    pred.to_csv(f'{obj}_{optimizer_name}_{lag}_{net}_pred.csv')
    
    return record
    

#%%

if __name__ == '__main__':
    
    data = pd.read_csv(r'training_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.iloc[:,0:1]
    
    # clipvalue = 0.5
    obj = 'CPI'
    Adam = optimizers.Adam()
    RMSprop = optimizers.RMSprop()
    SGD = optimizers.SGD()
    Adamax = optimizers.Adamax()
    optimizers_list = ['Adam', 'RMSprop', 'SGD']
    optimizers_dic = {'Adam':Adam, 'RMSprop':RMSprop, 'SGD':SGD, 'Adamax':Adamax}
    
    lags = [12, 24, 48]
    nets = [32, 64, 128]

    records = pd.DataFrame(columns=['Optimizer', 'Lag size', 'Neurons', 'MSE', 'RMSE', 'MSE_pred', 'RMSE_pred'])
    
    
    for optimizer in optimizers_list:
        for lag in lags:
            for net in nets:
                scaler, x, y, x_train, x_test, y_train, y_test = data_processing(data, obj, lag)
                record = lstm(scaler, x, y, x_train, x_test, y_train, y_test, obj, 
                              optimizers_dic[optimizer], optimizer, lag, net)
                records = pd.concat([records, record])
    
    
    records = records.reset_index(drop=True)
    records.to_csv(f'{obj}_1.csv')

# %%
