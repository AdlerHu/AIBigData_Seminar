# multivariate multi-step encoder-decoder lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

import MySQLdb
import pandas as pd
import sys


# 連接資料庫
def connect_database():
    db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')
    cursor = db.cursor()
    db.autocommit(True)
    return db, cursor


# 從資料庫讀出資料
# 時間限定讓筆數 30n +1
def load_data():
    db, cursor = connect_database()

    sql = '''select amount,p_banana_d1_amount,d5_avg_price 
             from Prediction_Source
             where (market_no = '830') and (trade_date between '2012-02-23' and '2020-07-01');'''

    cursor.execute(sql)
    dataset = pd.read_sql_query(sql, db).astype('float32')

    return dataset


# 自訂計算 MAPE 的函數
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.00001))) * 100


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard months
    train, test = data[:1980], data[1980:]

    print(len(train))
    print(len(test))

    # 資料正規化 (使用四分位距標準化資料)
    # sc = RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=False,
    #                   with_scaling=True)

    # sc = MaxAbsScaler(copy=True)

    # sc = Normalizer(norm='max', copy=True)

    sc = StandardScaler(copy=True, with_mean=True, with_std=False)

    train = sc.fit_transform(train)
    test = sc.fit_transform(test)

    # restructure into windows of monthly data
    train = array(split(train, len(train) / 30))
    test = array(split(test, len(test) / 30))

    test_groups = test.shape[0]  # 測試集 smaple數(以30天為單位)

    return train, test, test_groups


# evaluate a single model
def evaluate_model(test_groups, train, test, n_input):
    # fit model
    model = build_model(train, n_input)

    # past is a list of monthly data
    past = [x for x in train]

    # walk-forward validation over each month
    predictions = list()
    for i in range(len(test)):
        # predict the month
        yhat_sequence = forecast(model, past, n_input)

        # store the predictions
        predictions.append(yhat_sequence)

        # get real observation and add to past for predicting the next month
        past.append(test[i, :])

    # evaluate predictions days for each month
    predictions = array(predictions)

    reality = test[:, :, 0]

    # test[:, :, 0] 是實際值, predictions 是預測值
    score, scores = evaluate_forecasts(test_groups, reality, predictions)

    return score, scores, model


def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)

    # define parameters
    verbose, epochs, batch_size = 2, 150, 10
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.summary()

    # compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # 當監測值不再改善時，如發現損失沒有下降，則經過 patience 次 epoch 後停止訓練。
    estop = EarlyStopping(monitor='loss', patience=40)
    # fit model

    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_split=0.1,  # 使用訓練集的 1成 做為驗證集
                        callbacks=[estop])

    # 要轉換的歷史記錄  EMA 係數
    def to_EMA(points, a=0.1):
        ret = []  # 儲存轉換結果的串列
        EMA = points[0]  # 第 0 個 EMA 值
        for pt in points:
            EMA = pt * a + EMA * (1 - a)  # 本期EMA = 本期值*0.3 + 前期EMA * 0.7
            ret.append(EMA)  # 將本期EMA加入串列中
        return ret

    # 顯示 loss 學習結果
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    pyplot.plot(range(len(loss)), loss, marker='.', label='loss(train)')
    pyplot.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(validation)')
    pyplot.legend(loc='best')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss (mse)')
    pyplot.show()

    # # 顯示 mae 學習結果
    # mae = history.history['mae']
    # val_mae = history.history['val_mae']  # 取得驗證的歷史紀錄
    # idx = np.argmin(val_mae)  # 找出最佳(低)驗證週期
    # val = val_mae[idx]  # 取得最佳(低)驗證的val_mae值
    #
    # his_EMA = to_EMA(history.history['val_mae'])  # 將 val_mae 的值轉成 EMA 值
    # idx = np.argmin(his_EMA)  # 找出最低 EMA 值的索引
    # ema_val = his_EMA[idx]
    # print('最小 EMA 為第', idx + 1, '週期的', his_EMA[idx])
    #
    # mylog1 = open('LSTM_A_Fongshan(15to30)2_ema.log', mode='a+', encoding='utf-8')
    # print(f'2-2 最小 EMA 為第 {idx + 1} 週期的 {his_EMA[idx]}', file=mylog1)
    # mylog1.close()
    #
    # pyplot.plot(range(len(loss)), mae, marker='.', label='loss(train)')
    # pyplot.plot(range(len(val_mae)), val_mae, marker='.', label='val_loss(validation)')
    # pyplot.legend(loc='best')
    # pyplot.xlabel('epoch')
    # pyplot.ylabel('loss (mae)')
    # # pyplot.ylim([0.00001, 10])  # y軸邊界
    # pyplot.title(f'Best val_mae at epoch = {idx + 1} val_mae={ema_val:.3f}')
    # pyplot.show()

    return model


# convert past into inputs and outputs
def to_supervised(train, n_input, n_out=30):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()

    in_start = 0
    # step over the entire past one time step at a time
    for _ in range(len(data)):

        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out

        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])

        # move along one time step
        in_start += 1

    return array(X), array(y)


# make a forecast
def forecast(model, past, n_input):
    # flatten data
    data = array(past)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

    # retrieve last observations for input data
    input_x = data[-n_input:, :]

    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))

    # forecast the next month by 30-day
    yhat = model.predict(input_x, verbose=0)

    # we only want the vector forecast
    yhat = yhat[0]

    return yhat


# evaluate one or more monthly forecasts against expected values
def evaluate_forecasts(test_groups, actual, predicted):
    scores = list()
    # calculate an RMSE score for each day

    pred = predicted.reshape((test_groups, 30))

    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)

        # store
        scores.append(rmse)

    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    # mse = score ** 2

    mape = mean_absolute_percentage_error(actual, pred)
    print(f'MAPE: {mape:.3f}')
    print(f'MSE: {score ** 2:.3f}')
    print(f'RMSE: {score:.3f}')

    mylog = open('LSTM_A_Fongshan(15to30)2.log', mode='a+', encoding='utf-8')
    print(f'2-1 MAPE: {mape:.3f} MSE: {score ** 2:.3f} RMSE: {score:.3f}', file=mylog)
    mylog.close()

    #   print(actual)
    #   print(pred)

    pyplot.title('1-month Amount Prediction for Fongshan')
    pyplot.plot(actual[-1], label='real amount')
    pyplot.plot(pred[-1], label='predicted amount')
    pyplot.legend(loc='best')
    pyplot.xlabel('Day')
    pyplot.ylabel('Amount')
    pyplot.savefig('Fongshan(15to30)amount, price, d1_price, d1_origin_price, p_banana_d1_price2-1')
    pyplot.show()

    return score, scores


# summarize RMSE scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.3f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def predict_future(dataset, model, n_input):
    # 取最近n_input天資料
    last_n_input = dataset.iloc[-n_input:, :].values

    # 製作 samples
    # 改維度: 2D -> 3D
    last_n_input_3D = np.reshape(last_n_input, (1, n_input, last_n_input.shape[1]))

    # 預測未來量
    future = model.predict(last_n_input_3D)
    print(future.ndim, type(future), future.shape)
    print('Predicted Amount for 30-Day:')
    print(future[0])


def main():
    dataset = load_data()

    # split into train and test
    train, test, test_groups = split_dataset(dataset.values)

    # evaluate model and get scores
    n_input = 15
    score, scores, model = evaluate_model(test_groups, train, test, n_input)

    # summarize RMSE scores
    summarize_scores('Fongshan LSTM Model RMSE score', score, scores)

    # plot scores
    days = [x for x in range(1, 31)]
    pyplot.plot(days, scores, marker='.', label='lstm')
    pyplot.xlim([0, 31])  # x軸邊界
    # pyplot.ylim([0.0001,2])  # y軸邊界
    pyplot.grid(True, linestyle="--", color='gray', linewidth='0.5', axis='both')
    pyplot.legend(loc='best')
    pyplot.xlabel('Day')
    pyplot.ylabel('RMSE')
    pyplot.title('LSTM Model Prediction Score')
    pyplot.savefig('FongshanA(15to30)amount, price, d1_price, d1_origin_price, p_banana_d1_price_score2-1')
    pyplot.show()

    predict_future(dataset, model, n_input)

    # 存初始權重
    wt = model.get_weights()

    # 存儲模型與權重
    model.save('Fongshan.h5')

    del model


if __name__ == '__main__':
    main()
