import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import MySQLdb
import pandas as pd
import csv


# 連接資料庫
def connect_database():
    db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')
    cursor = db.cursor()
    db.autocommit(True)
    return db, cursor


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 自訂計算 MAPE 的函數
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.00001))) * 100


# 把資料轉換成可以機器學習的形式
def frame_data_as_supervised_learning(scaler, values, veri_values, columns):
    scaled = scaler.fit_transform(values)
    veri_scaled = scaler.fit_transform(veri_values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    veri_reframed = series_to_supervised(veri_scaled, 1, 1)

    # print(reframed.describe())
    drop_columns(reframed, veri_reframed, columns)

    return reframed, veri_reframed


# 刪除不要的欄位
# 資料有 n 欄, drop 掉 n+1 到 2n-1
#        8 欄  drop    9, 10, 11, 12, 13, 14, 15
def drop_columns(reframed, veri_reframed, columns):
    list1, list2 = [], []
    for i in range(columns + 1, (columns + 1) * 2 - 2):
        list1.append(i)

    list2.append(list1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[list2], axis=1, inplace=True)
    veri_reframed.drop(veri_reframed.columns[list2], axis=1, inplace=True)

    return reframed, veri_reframed


# 用訓練天數切出訓練、測試資料集
def split_data_set(train_days, reframed, veri_reframed):
    # split into train and test sets
    values = reframed.values
    veri_values = veri_reframed.values

    train = values[:train_days, :]
    test = values[train_days:, :]
    verification = veri_values[:, :]

    return train, test, verification


# 切出x, y並改變形狀
def split_and_reshape(train, test, verification):
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    veri_X, veri_y = verification[:, :-1], verification[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    veri_X = veri_X.reshape((veri_X.shape[0], 1, veri_X.shape[1]))

    return train_X, train_y, test_X, test_y, veri_X, veri_y


# 有出形狀不對的錯誤時候用這個函數檢查
def check_shape(train_X, train_y, test_X, test_y, veri_X, veri_y):
    print(f'trainX: {train_X.shape}')
    print(f'trainY: {train_y.shape}')
    print(f'testX: {test_X.shape}')
    print(f'testY: {test_y.shape}')
    print(f'veriX: {veri_X.shape}')
    print(f'veriY: {veri_y.shape}')


# 用train好的模型做預測
def predict(scaler, model, veri_X, veri_y):
    # make a prediction
    yhat = model.predict(veri_X)
    veri_X = veri_X.reshape((veri_X.shape[0], veri_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, veri_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    veri_y = veri_y.reshape((len(veri_y), 1))
    inv_y = concatenate((veri_y, veri_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # for i in range(inv_yhat.size):
    #     # print(inv_y[i] - inv_yhat[i])
    #     print(f'{inv_y[i]}, {inv_yhat[i]}')

    return inv_y, inv_yhat


# 驗證模型的準確度
def evaluate_model(inv_y, inv_yhat, variable):
    # caculate MAPE
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    # print('Verification MAPE: %.3f' % mape)

    # caculate MSE
    mse = mean_squared_error(inv_y, inv_yhat)
    # print('Verification MSE: %.3f' % mse)

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Verification RMSE: %.3f' % rmse)

    print(variable)
    print('%.3f, %.3f, %.3f' % (mape, mse, rmse))

    with open('price/Fongshan_price.csv', 'a', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow([variable, '%.3f' % mape, '%.3f' % mse, '%.3f' % rmse])

    # pyplot.plot(inv_y, label='real')
    # pyplot.plot(inv_yhat, label='prediction')
    # pyplot.legend()
    # pyplot.show()


# 從資料庫讀取出要用的欄位
# !! 改這裡 !!
# 要預測的放在第一欄
def load_data(db, cursor):

    train_sql = """SELECT price, amount, d1_price FROM Prediction_Source
    WHERE (market_no = '830') and (trade_date between '2012-01-01' and '2020-04-30');"""

    veri_sql = """SELECT price, amount, d1_price FROM Prediction_Source
    WHERE (market_no = '830') and (trade_date between '2020-04-30' and '2020-05-31');"""

    cursor.execute(train_sql)

    dataset = pd.read_sql_query(train_sql, db)
    veri_set = pd.read_sql_query(veri_sql, db)

    values = dataset.values
    veri_values = veri_set.values

    columns = values.shape[1]

    variable = train_sql[train_sql.index('price'):train_sql.index('FROM')]
    return values, veri_values, columns, variable


# model
# !! 改這裡 !!
def train_model(train_X, train_y, test_X, test_y):
    # design network
    model = Sequential()
    model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_y, epochs=200, batch_size=20, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()
    return model, history


def main():
    db, cursor = connect_database()

    values, veri_values, columns, variable = load_data(db, cursor)

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    reframed, veri_reframed = frame_data_as_supervised_learning(scaler, values, veri_values, columns)

    # 訓練天數, 交易天數少的市場要改
    train_days = 1000

    train, test, verification = split_data_set(train_days, reframed, veri_reframed)

    train_X, train_y, test_X, test_y, veri_X, veri_y = split_and_reshape(train, test, verification)

    # print_shape(train_X, train_y, test_X, test_y, veri_X, veri_y)

    model, history = train_model(train_X, train_y, test_X, test_y)

    print('-------------------------------------------------------')

    inv_y, inv_yhat = predict(scaler, model, veri_X, veri_y)

    evaluate_model(inv_y, inv_yhat, variable)


if __name__ == '__main__':
    main()
