# univariate multi-step lstm
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
import numpy as np


# 自訂計算 MAPE 的函數
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.00001))) * 100


# split a univariate dataset into train/test sets
def split_dataset(data):
    idfk = 1000

    # split into standard weeks
    # train, test = data[1:-328], data[-328:-6]
    train, test = data[1:901], data[901:-2]

    # train, test = data[1:idfk], data[idfk:-6]

    # print(len(train))
    # print(len(test))

    # 除30 會失敗
    # 因為除不盡

    # restructure into windows of weekly data
    train = array(split(train, len(train) / 30))
    test = array(split(test, len(test) / 30))

    print(f'train.shape: {train.shape[2]}')
    print(f'test.shape: {test.shape[2]}')
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
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
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    # print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=30):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))

    print('%%%%%%%%%%%%%%%%')
    print(f'train: {train.shape}')
    print('%%%%%%%%%%%%%%%%')

    print('%%%%%%%%%%%%%%%%')
    print(f'data: {data.shape}')
    print('%%%%%%%%%%%%%%%%')

    # Train 的 x, y
    X, y = list(), list()

    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1

    return array(X), array(y)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)

    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print(f'train_x: {train_x.shape}')
    print(f'train_y: {train_y.shape}')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    # define parameters
    verbose, epochs, batch_size = 2, 10, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    print(f'n_timesteps: {n_timesteps}, n_features: {n_features},  n_outputs: {n_outputs}')

    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

    # print('################')
    # print(data.shape)
    # print('################')

    # retrieve last observations for input data
    input_x = data[-n_input:, 0]

    # print('################')
    # print(input_x.shape)
    # print('################')

    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))

    # print('################')
    # print(input_x)
    # print('################')


    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]

    print('******************')
    print(history)
    print('******************')

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

    # test[:, :, 0] 似乎是實際值， predictions 似乎是預測值
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)

    # print('$$$$$$$$$$$$$$$$')
    # print(test[:, :, 0])
    # print(len(test[:, :, 0]))
    # print('$$$$$$$$$$$$$$$$')
    #
    # print(predictions)
    # print(len(predictions))
    # print('$$$$$$$$$$$$$$$$')

    mape = mean_absolute_percentage_error(test[:, :, 0], predictions)
    # print(f'MAPE: {mape}')

    return score, scores, model


# ----------------------------------------------------------------------------------------------------------

# load the new file
# dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True,
#                    parse_dates=['datetime'], index_col=['datetime'])

dataset2 = read_csv('damn.csv', header=0, infer_datetime_format=True,
                    parse_dates=['trade_date'], index_col=['trade_date'])

# split into train and test
train, test = split_dataset(dataset2.values)

# print("$$$$$$$$$")
# print(test.shape)
# print("$$$$$$$$$")

# evaluate model and get scores
# n_input = 7
n_input = 10
score, scores, model = evaluate_model(train, test, n_input)

# summarize scores
summarize_scores('lstm', score, scores)

# plot scores
# days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']

days = [x for x in range(0, 30)]

pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# 取最近10天價格
last_10_day = dataset2.iloc[-n_input:, 1:].values.astype('float32')
#print('last_seven_day:\n', last_14_day, '\n')

print('@@@@@@@@@@@@@@@@@@')
# print(last_10_day)
print('last_10_day筆數:', len(last_10_day))
print('類別:', last_10_day.__class__, ';', '陣列形狀:', last_10_day.shape, '陣列維度:', last_10_day.ndim)


# 製作 samples

# 改維度: 2D -> 3D
last_10_day_3D = np.reshape(last_10_day, (1, n_input, 7))
# print(last_10_day_3D.shape)

# print('last_14_day_3D:', len(last_14_day_3D))
# print('類別:', last_14_day_3D.__class__, ';', '陣列形狀:', last_14_day_3D.shape, '陣列維度:', last_14_day_3D.ndim, '\n')

# future = model.predict(last_10_day_3D)
# print(future)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


