# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:20:28 2023

@author: chai_l
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

#%%

def data_processing(step, datas, factor, obj, lag):
    
    if factor == 1:
        data = datas[[obj]]
    elif factor == 2:
        data = datas.iloc[:,0:2]
    else:
        data = datas
    
    dim = data.shape[1]

    scaler = MinMaxScaler(feature_range=(0, 1))

    def datareshape(data, obj, lag, step, dim):

        values = data.values
        forecast = data[obj]

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
# lstm

def lstm(scaler, x, y, x_train, x_test, y_train, y_test, obj, optimizer, lag, net):

    batch_size = 60
    epochs = 500

    model = Sequential()
    model.add(LSTM(units=net, activation='relu', return_sequences=True, input_shape=(x.shape[1],x.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units=net, activation='relu', return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1]*2))
    model.add(Dense(y_train.shape[1]))
    print(model.summary())

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    callback = EarlyStopping(monitor='val_loss',patience=15, restore_best_weights=True)
    History = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2,
                        callbacks=[callback])

    model.fit(x_train, y_train, validation_split=0.2, callbacks=[callback], epochs=epochs)
    
    fig_train = plt.figure()
    plt.plot(History.history['loss'], label='loss')
    plt.plot(History.history['val_loss'], label = 'val loss')
    plt.legend(loc='best')
    plt.savefig(f'forward_{y_train.shape[1]}_{obj}_{optimizer}_{lag}_{net}_loss.png')
    plt.show()

    
    y_pred_norm = model.predict(x_test)
   

    y_pred = scaler.inverse_transform(y_pred_norm)
    y_test = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    
    
    y_true = y[-round(x.shape[0]*0.2):]
    xx_test = x[-round(x.shape[0]*0.2):]
    yy_pred_norm = model.predict(xx_test)
    yy_pred = scaler.inverse_transform(yy_pred_norm)
    
    mse_pred = mean_squared_error(y_true, yy_pred)
    rmse_pred = np.sqrt(mse_pred)
    print(f'MSE_pred: {mse_pred}')
    print(f'RMSE_pred: {rmse_pred}')
      
    
    # record results
    record = pd.DataFrame({'Optimizer':[optimizer], 'Lag size':[lag], 'Neurons':[net], 'MSE':[mse], 
              'RMSE':[rmse], 'MSE_pred':[mse_pred], 'RMSE_pred':[rmse_pred]})
    
    #save raw data
    test = pd.DataFrame(np.zeros([y_test.shape[0], y_test.shape[1]*2]), 
                        columns=pd.MultiIndex.from_product([['test','pred'],[1,2,3]]))
    
    pred = pd.DataFrame(np.zeros([y_true.shape[0], y_true.shape[1]*2]), 
                        columns=pd.MultiIndex.from_product([['true','pred'],[1,2,3]]))
    
    test['test'] = y_test
    test['pred'] = y_pred
    pred['true'] = y_true
    pred['pred'] = yy_pred
    
    test.to_csv(f'forward_{y_pred.shape[1]}_{obj}_{optimizer}_{lag}_{net}_test.csv')
    pred.to_csv(f'forward_{yy_pred.shape[1]}_{obj}_{optimizer}_{lag}_{net}_pred.csv')
    
    
    return record, y_test, y_pred, y_true, yy_pred

#%%
# visualisation
def visualisation(data, obj, y_test, y_pred, y_true, yy_pred):

    in_pred = pd.DataFrame(yy_pred[-y_pred.shape[1]-1], index=data.index[-y_pred.shape[1]:], columns=['Prediction'])
    
    time = pd.date_range(data.index[-2], periods=y_pred.shape[1]+1, freq='m')
    time = time + dt.timedelta(days=1)
    out_value = np.r_[yy_pred[-y_pred.shape[1]-1][-1], yy_pred[-1]]
    out_pred = pd.DataFrame(out_value, index=time, columns=['Prediction_forward'])
   
    
    fig_test = plt.figure(figsize=(10, 5))
    ax_pre = plt.subplot(1,1,1)
    ax_pre.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.plot(data[obj][-24:], label='True value', marker='o')
    plt.plot(in_pred, label='Prediction', linestyle = '--', marker='o', color='darkorange', alpha=0.8)
    plt.plot(out_pred, label='Forward prediction', linestyle = '--', marker='o', color='red', alpha=0.8)
    plt.legend(loc='best')
    plt.ylabel('%')
    plt.savefig(f'forward_{y_pred.shape[1]}_{obj}_{optimizer}_{lag}_{net}_pred.png')
    plt.show()


#%%
if __name__ == '__main__':
    
    datas = pd.read_csv(r'C:\Users\chai_l\Downloads\Lei Chai\training_data.csv')
    datas['Date'] = pd.to_datetime(datas['Date'])
    datas.set_index('Date', inplace=True)
    
    step = 3
    
    objs = ['CPI', 'CPI', 'Unemployment', 'Unemployment']
    factors = [1, 2, 5, 5]
    optimizer = 'Adam'
    lags = [24, 24, 24, 48]
    nets = [32, 128, 64, 128]
    
    records = pd.DataFrame(columns=['Optimizer', 'Lag size', 'Neurons', 'MSE', 'RMSE', 'MSE_pred', 'RMSE_pred'])
    
    for obj, factor, lag, net in zip(objs, factors, lags, nets):
        scaler, x, y, x_train, x_test, y_train, y_test = data_processing(step, datas, factor, obj, lag)
        record, y_test, y_pred, y_true, yy_pred = lstm(scaler, x, y, x_train, x_test, y_train, y_test, obj, optimizer, lag, net)
        visualisation(datas, obj, y_test, y_pred, y_true, yy_pred)
        records = pd.concat([records, record])
        
    records = records.reset_index(drop=True)
    records.to_csv(f'forward_{step}_{obj}.csv')
    
    
    
