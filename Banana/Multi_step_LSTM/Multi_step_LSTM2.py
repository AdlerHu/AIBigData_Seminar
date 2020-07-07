# 從 CSV 檔讀取變數,進行變數效果測試
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
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import MySQLdb
import pandas as pd
import csv
from keras.callbacks import EarlyStopping


# !! 改這裡 !!
# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    # train, test = data[1:-328], data[-328:-6]
    train, test = data[:1980], data[1980:]

    print(len(train))
    print(len(test))

    # restructure into windows of weekly data
    train = array(split(train, len(train) / 30))
    test = array(split(test, len(test) / 30))

    test_groups = test.shape[0]

    return train, test, test_groups


# !! 改這裡 !!
# 從 CSV 檔讀取變數的函式
def load_variables():
    variables = []
    with open('price/Variables.csv', newline='') as csvfile:
        # 讀取 CSV 檔案內容
        data = csv.reader(csvfile)

        for row in data:
            variables.append(row[0])
    return variables


# !! 改這裡 !!
# 將模型指標寫成 CSV 檔的函式
def result(variable, mape, rmse):
    mse = rmse ** 2
    with open('price/Taipei2.csv', 'a', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow([variable, '%.3f' % mape, '%.3f' % mse, '%.3f' % rmse])


# !! 改這裡 !!
# 從資料庫讀出資料
# 時間限定讓筆數 30n +1
def load_data(variable):
    db, cursor = connect_database()

    sql = f"select {variable} from Prediction_Source " \
          f"where (market_no = '104') and (trade_date between '2012-01-11' and '2020-07-01'); "

    cursor.execute(sql)
    dataset = pd.read_sql_query(sql, db)

    return dataset


# !! 改這裡 !!
# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 2, 100, 10
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mae', optimizer='adam')

    estop = EarlyStopping(monitor='loss', patience=60)

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[estop])
    return model


# 連接資料庫
def connect_database():
    db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')
    cursor = db.cursor()
    db.autocommit(True)
    return db, cursor


# 自訂計算 MAPE 的函數
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.00001))) * 100


# evaluate a single model
def evaluate_model(variable, test_groups, train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)

        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)

    reality = test[:, :, 0]

    # test[:, :, 0] 是實際值, predictions 是預測值
    score, scores = evaluate_forecasts(variable, test_groups, reality, predictions)

    return score, scores, model


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=30):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
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
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(variable, test_groups, actual, predicted):
    scores = list()
    # calculate an rmse score for each day

    pred = predicted.reshape((test_groups, 30))

    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # print(actual.shape)
    # print('-----------')
    # print(pred.shape)
    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)

        # store
        scores.append(rmse)
    # calculate overall rmse
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))

    mape = mean_absolute_percentage_error(actual, pred)
    print(f'MAPE: {mape}')

    result(variable=variable, mape=mape, rmse=rmse)
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def predict_future(dataset, model, n_input):
    # 取最近10天資料
    last_10_day = dataset.iloc[-n_input:, :].values.astype('float32')

    # print('@@@@@@@@@@@@@@@@@@')
    # print(f'last_10_day.shpae: {last_10_day.shape}')

    # 製作 samples
    # 改維度: 2D -> 3D
    last_10_day_3D = np.reshape(last_10_day, (1, n_input, last_10_day.shape[1]))
    # print(f'last_10_day_3D.shape: {last_10_day_3D.shape}')

    future = model.predict(last_10_day_3D)
    print(future[0])


def main():

    variables = load_variables()

    for variable in variables:

        dataset = load_data(variable=variable)

        # split into train and test
        train, test, test_groups = split_dataset(data=dataset.values)

        # evaluate model and get scores
        n_input = 10
        score, scores, model = evaluate_model(variable=variable, test_groups=test_groups,
                                              train=train, test=test, n_input=n_input)

        # summarize scores
        summarize_scores(name='lstm', score=score, scores=scores)

        # plot scores
        days = [x for x in range(0, 30)]
        # pyplot.plot(days, scores, marker='o', label='lstm')
        # pyplot.show()

        predict_future(dataset=dataset, model=model, n_input=n_input)


if __name__ == '__main__':
    main()
