import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # 資料最小最大值標準化(MinMaxScaler)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import tensorflow as tf
import keras
import math

# 2. 數據集拆分 -- 訓練集 & 測試集

dataset = pd.read_csv(r'./p_factors_109_test.csv', usecols=[1, 2],
                      engine='python')

# 欄位型別轉換:原 float64 -> float32
dataset = dataset.astype('float32')
# print(dataset.info())

# 預測點y 的前 rolling_time 天的資料
rolling_time = 14

n = len(dataset) - rolling_time

# 訓練集: 從第一筆 到 第n筆
dataset_train = dataset.iloc[:n]
# 測試集: 第n筆 到 最後一筆
dataset_test = dataset.iloc[n:]

# 設定隨機種子
seed = 7
np.random.seed(seed)

# 3. 製作 訓練 數據集

# 訓練集 正規化 (將資料特徵縮放至[0, 1]間)
# 資料縮放
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(dataset_train)
test_set_scaled = sc.fit_transform(dataset_test)

print(f'Train Shape: {training_set_scaled.shape}')
print(f'Test Shape: {test_set_scaled.shape}')

# # ++++++++++++++++++++++注意：to get the original scale 有問題 ++++++++++++++++++++++
dataset_train = sc.inverse_transform(training_set_scaled)
# print(dataset_train)

# # ++++++++++++++++++++++注意：to get the original scale 有問題 ++++++++++++++++++++++
test_set = sc.inverse_transform(test_set_scaled)
# print(test_set)

# 讀取 訓練集欄位「price」,「amount」的真實值 for 畫圖比對用
# 語法: <dataframe>.iloc[row:row, col:col].values
real_price = dataset_test.iloc[:, 0:1].values
real_amount = dataset_test.iloc[:, 1:2].values

# 欄位型別轉換:原 float64 -> float32，語法：<dataframe>.astype('float32')
real_price = real_price.astype('float32')
real_amount = real_amount.astype('float32')

print(f'Real_price Shape: {real_price.shape}')
print(f'Real_amount Shape: {real_amount.shape}')

# 5. 製作訓練集 samples

# 預測點y 的前 rolling_time 天的資料
X_train = []  # 預測點的前 rolling_time 天的資料
# y_train = []  # 預測點 (用 前 rolling_time天 預測 未來 predicted_day 天)
y_shit = []
y_train = ()  # 預測點 (用 前 rolling_time天 預測 未來 predicted_day 天)

predicted_day = 7

for i in range(rolling_time, len(dataset_train)):
    if (i + predicted_day) <= len(dataset_train):
        X_train.append(training_set_scaled[(i - rolling_time):i, :])
        # y_train.append(training_set_scaled[i:(i + predicted_day), :])

        y_shit.append(training_set_scaled[i:(i + predicted_day), :])
        y_train = tuple(y_shit)
    if (i + predicted_day) > len(dataset_train):
        break

# 6's summarization
print('X_train 訓練集samples數:', len(X_train))
print('y_train 訓練集samples數:', len(y_train))

# # 7. 改變 訓練集 samples 類別： 從 list  變成 array
# 將 X_train, y_train 這二個list，各別轉成 numpy array的格式，以利輸入 RNN
X_train_2D, y_train_2D = np.array(X_train), np.array(y_train)
# X_train_2D = np.array(X_train)
# y_train_2D = np.array(y_train).reshape([len(y_train), 7])

print('--------------------')
print(f'X_train_2D type: {type(X_train_2D)}')
print(f'X_train_2D type: {type(y_train_2D)}')
print(f'X_train_2D shape: {X_train_2D.shape}')
print(f'X_train_2D shape: {y_train_2D.shape}')
print(f'y_train_2D ndim: {y_train_2D.ndim}')
print(f'y_train_2D ndim: {X_train_2D.ndim}')

# 7's summarization
print('#7. 改變 訓練集 samples 類別： 從 list  變成 array:\n')
print(f'X_train_2D shape: {X_train_2D.shape}')
print(f'y_train_2D shape: {y_train_2D.shape}')
print(f'y_train_2d ndim: {y_train_2D.ndim}')

# 8. 改變訓練集 samples 的矩陣形狀:  二維 －> 三維,以符合Keras之RNN－LSTM輸入層的資料格式

# 首先我們會可以用一個最為單純的LSTM(Vanilla LSTM)來進行預測，視為BaseLine。
# Vanilla LSTM的model架構如下所示：
# *************************************************************************
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# *************************************************************************

# 因為現在 X_train_2D 是 2-dimension，將它 reshape 成 3-dimension: [ samples, timesteps, features ]
#
# input_shape是輸入資料格式，LSTM 輸入層資料格式: 為3維矩陣, 矩陣內容 [ samples, timesteps, features ]
# *******************************************
# samples:   訓練集 samples 的數量
# timesteps: 每個 sample 裡有幾個時間步數(時間視窗)－> how many timesteps the LSTM will learn from each samlpe
# features:  每個 sample 含多少個 特徵(欄位數)
# *******************************************

# --------訓練集 samples array改維度-------

# 取二個欄位price, amount
features = 2

# <numpy.ndarray>.shape[0] 是取矩陣第一维度的長度
# <numpy.ndarray>.shape[1] 是取矩陣第二维度的長度
# input_shape是3維矩陣，並且矩陣形狀，需為 (samples數, timesteps數, features數)
input_shape = (X_train_2D.shape[0], X_train_2D.shape[1], features)

# 2維矩陣 reshape 3維矩陣，語法: np.reshape(舊矩陣, 新矩陣形狀)
# 《注意》 因為轉換前後的元素數目一樣，所以能夠成功的進行轉換，假如前後數目不一樣的話，則會有 ValueError。
X_train_3D = np.reshape(X_train_2D, input_shape)

# summarization of #8
print('#8. 改變訓練集 samples 的矩陣形狀:  二維 －> 三維:\n')
print('X_train_3D 觀測值筆數:', len(X_train_3D))
print('陣列類別:', X_train_3D.__class__, ';', '陣列形狀:', X_train_3D.shape, '陣列維度:', X_train_3D.ndim, '\n')

# # #  9. 建立及訓練 LSTM 模型
#
# # 搭建 LSTM layer: units: 神經元的數目
# #
# # 第一層(輸入層)的 LSTM Layer 記得要設定input_shape參數 搭配使用dropout，這裡設為 0.2
# # 第四層(隱藏層) LSTM Layer 即將跟 Ouput Layer 做連接，因此注意這裡的 return_sequences 設為預設值 False （也就是不用寫上 return_sequences）
# # 為了構建LSTM，我們需要從Keras中匯入幾個模組:
# #
# # Sequential： 用於初始化神經網路
# # Dense：      用於新增密集連線的神經網路層
# # LSTM：       用於新增長短期記憶體層
# # Dropout：    用於新增防止過擬合的dropout層
# # units:       神經元的數目

# # -------- 輸入層 --------
tf.model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# Input is shape ( the number of timesteps x the number of  timeseries) -> (14, 2)
# output is shape (the number of  timesteps x the number of variables) -> (7, 32) because return_sequences=True
tf.model.add(LSTM(units=100, activation='relu', input_shape=(X_train_3D.shape[1], features), return_sequences=True))
# tf.model.add(Dropout(0.2))
#
# # -------- 隱藏層 (核心層)--------
#
# # 1. Adding a second LSTM layer and some Dropout regularisation
# tf.model.add(LSTM(units=100, return_sequences=True))
# tf.model.add(Dropout(0.2))
#
# # 2. Adding a third LSTM layer and some Dropout regularisation
# tf.model.add(LSTM(units=100, return_sequences=True))
# tf.model.add(Dropout(0.2))
#
# # 3. Adding a fourth LSTM layer and some Dropout regularisation
# tf.model.add(LSTM(units=100))
# tf.model.add(Dropout(0.2))

# -------- 輸出層 --------

# # 因爲這是一個迴歸問題，不需要將預測結果進行分類轉換，所以輸出層不設置激活函數，直接輸出數值。
# # https://www.twblogs.net/a/5c3b7a2cbd9eee35b21dd536
# tf.model.add(Dense(units=features))

# # 顯示出模型摘要
tf.model.summary()

# -------- 定義優化方式 --------
# Compiling
tf.model.compile(optimizer='adam', loss='mean_squared_error',
                 metrics=['accuracy'])  # 'accuracy' 為 需要觀察的資訊

# 將 epoch 的訓練結果保存在 csv 文件中
logger = CSVLogger('model_KerasLSTM_109_Hero_14day_muti2.log')

# 當監測值 val_loss 不再改善時，如發現損失沒有下降，則經過 patience 次 epoch 後停止訓練。
estop = EarlyStopping(monitor='val_loss', patience=100)

# -------- 訓練與評估模型 --------

# 模型訓練
epochs_time = 500

tensorboard_callback = [
    tf.keras.callbacks.EarlyStopping(patience=100),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

tf.model.fit(X_train_3D, y_train_2D,  # 進行訓練的因和果資料
             epochs=epochs_time,  # epochs：把訓練樣本丟進神經網絡訓練 epochs_time 次，也就是機器學習的次數
             batch_size=10,  # batch_size： 每次丟進訓練的樣本數，即設定每次訓練的筆數
             verbose=2,  # verbose：日誌顯示
             # verbose = 0 為不在標準輸出流輸出日誌信息
             # verbose = 1 為輸出進度條記錄 # 預設
             # verbose = 2 為每個 epoch 輸出一行記錄
             validation_split=0.1,  # validation_split 0~1 之間的浮點數，用來指定訓練集 當作驗證集的比例
             shuffle=True,
             callbacks=[logger, estop, tensorboard_callback])  # 每筆數裡呼叫後儲存在logger, estop, tensorboard

# 進行模型評估
score = tf.model.evaluate(X_train_3D, y_train_2D, verbose=0)
print(score)
print('train loss:', score[0])
print('train acc:', score[1])

# # 10. 畫出學習曲線，保存模型參數
# 顯示 acc 學習結果

print(tf.model.history.history.keys())

accuracy = tf.model.history.history['accuracy']
val_accuracy = tf.model.history.history['val_accuracy']

plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示 loss 學習結果
loss = tf.model.history.history['loss']
val_loss = tf.model.history.history['val_loss']
plt.plot(range(len(loss)), loss, marker='.', label='loss(training data)')
plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# # # 讀取與保存模型權重格式為HDF5格式，需先安裝套件 h5py
# # get_ipython().system('pip install h5py')
#
# # # 存儲模型與權重
# # model.save('model_KerasLSTM_109_Hero_14day_muti.h5')
# # del model  # deletes the existing model
# #
# # # # 10. 製作測試集 samples:  split a univariate sequence into samples
# #
# # # 合併數據集(訓練集+測試集)
# # #dataset_total = pd.concat((dataset_train['price'], dataset_test['price']), axis = 0)
# # dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
# # print(type(dataset_total))
# # print(dataset_total)
# #
# # # 正規化
# # dataset_total_scaled = sc.fit_transform(dataset_total)
# # print(dataset_total_scaled)
# #
# #
# # # # 欄位型別轉換:原 float64 -> float32
# # # dataset_total = dataset_total.astype('float32')
# #
# # a = len(dataset_total) - len(dataset_test) - rolling_time
# # print('a:', a)
# #
# # # 讀取欄位值: 第a筆 到 最後一筆
# # inputs_2D = dataset_total_scaled[a :]
# # print('inputs 測試集筆數:', len(inputs_2D))
# # print('類別:', inputs_2D.__class__, ';', '陣列形狀:', inputs_2D.shape, '陣列維度:', inputs_2D.ndim, '\n')
# # print('inputs_2D:\n', inputs_2D)
# #
# # # # reshape 一維 轉 二維
# # # inputs_2D = inputs.reshape(-1,1)
# # # print('inputs_2D:\n', inputs_2D)
# # # print('inputs_2D 測試集筆數:', len(inputs_2D))
# # # print('類別:', inputs_2D.__class__, ';', 'inputs_2D陣列形狀:', inputs_2D.shape, 'inputs_2D陣列維度:', inputs_2D.ndim, '\n')
# # # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
# #
# # # # 資料正規化
# # # inputs_2D_scaled = sc.transform(inputs_2D) # Feature Scaling
# # # print('測試集 資料正規化:')
# # # print('inputs_2D_scaled 訓練集筆數:', len(inputs_2D_scaled))
# # # print('類別:', inputs_2D_scaled.__class__, ';', '陣列形狀:', inputs_2D_scaled.shape, '陣列維度:', inputs_2D_scaled.ndim, '\n')
# # # print('inputs_2D_scaled:\n', inputs_2D_scaled)
#
# # # 製作測試集 samples
# # X_test = []
# #
# # # dataset_test是 測試集
# # b = len(dataset_test) + rolling_time
# #
# # # numpy Array 取值 語法 array[row:row, col]
# # for i in range(rolling_time, b): # rolling_time=14, b=28
# #     X_test.append(inputs_2D[(i - rolling_time):i, :])
# # print('X_test 測試集 smaples數:', len(X_test))
# # print('類別:', X_test.__class__)
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # # --------測試集 samples 改類別-------
# #
# # # 將 X_test 這個list，轉成 numpy array的格式
# # X_test_2D = np.array(X_test)
# #
# # print('改變 測試集 samples 類別： 從 list  變成 array:')
# # print('X_test_2D 測試集smaples數:', len(X_test_2D))
# # print('類別:', X_test_2D.__class__, ';', '陣列形狀:',X_test_2D.shape, '陣列維度:', X_test_2D.ndim, '\n')
# # print('X_test_2D:\n', X_test_2D)
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # # --------測試集 samples array改維度: 2D -> 3D -------
# #
# # # 以符合輸入層要求的矩陣形狀，需為 (samples數, timesteps數, features數)
# # # <numpy.ndarray>.shape[0] 是取矩陣第一维度的長度
# # # <numpy.ndarray>.shape[1] 是取矩陣第二维度的長度
# # input_shape_X_test = (X_test_2D.shape[0], X_test_2D.shape[1], features)
# #
# # # 2D矩陣 reshape 3D矩陣，語法: np.reshape(舊矩陣, 新矩陣形狀)
# # # 《注意》 因為轉換前後的元素數目一樣，所以能夠成功的進行轉換，假如前後數目不一樣的話，則會有 ValueError。
# # X_test_3D = np.reshape(X_test_2D, input_shape_X_test)
# #
# # print('X_test_3D 測試集smaples數:', len(X_test_3D))
# # print('類別:', X_test_3D.__class__, ';', '陣列形狀:', X_test_3D.shape, '陣列維度:', X_test_3D.ndim, '\n')
# #
# # # # 11. 模型預測 y_test  歷史價格
#
# # # 載入模型
# # model = load_model('model_KerasLSTM_109_Hero_14day_muti.h5')
# #
# # # 顯示出模型摘要
# # model.summary()
#
# # # 丟X_test_3D值進去訓練好的模型，預測 y_test
# # predicted_y_test = model.predict(X_test_3D)
# #
# # # to get the original scale
# # predicted_y_test = sc.inverse_transform(predicted_y_test)
# #
# # predicted_y_test = predicted_y_test.reshape(-1,1)
# # print(type(predicted_y_test), predicted_y_test.shape, predicted_y_test.ndim)
# # # print(list(predicted_y_test))
# #
# # predicted_y_test_list =[]
# # for i in predicted_y_test:
# #     predicted_y_test_list.append(i[0])
# # #print(predicted_y_test_list)
# #
# # price, amount = [], []
# # for p in range(28):
# #     if p % 2 == 0:
# #         price.append(predicted_y_test_list[p])
# #     else:
# #         amount.append(predicted_y_test_list[p])
# # print('模型 預測 y_test 價格:\n', price)
# # print('模型 預測 y_test 量:\n', amount)
# #
# # # Visualising the results - Predicted Price for test
# # plt.plot(real_price, color = 'red', label = 'Real History Price')  # 紅線表示真實價格
# # plt.plot(price, color = 'green', label = 'Predicted history Price for test')  # 綠線表示預測價格
# # plt.title(str(rolling_time) + '-day Price forecast')
# # plt.xlabel('Day')
# # plt.ylabel('Avg Price')
# # plt.legend()
# # plt.show()
# #
# # plt.plot(real_amount, color = 'blue', label = 'Real History Amount')  # 紅線表示真實價格
# # plt.plot(amount, color = 'grey', label = 'Predicted history Amount for test')  # 綠線表示預測價格
# # plt.title(str(rolling_time) + '-day Amount forecast')
# # plt.xlabel('Day')
# # plt.ylabel('Trade Amount')
# # plt.legend()
# # plt.show()
# #
# # # # 13.  模型預測 未來1日 價格& 量
# #
# # # 讀取資料
# # dataset = pd.read_csv(r'C:\Users\fernh\workspace\EB101\Team Heart\LSTM\p_factors_109_test.csv', usecols=[1,2], engine='python')
# #
# # # 取最近14天價格
# # last_14_day = dataset.iloc[-rolling_time:, 0:2].values.astype('float32')
# # # print('last_seven_day:\n', last_14_day, '\n')
# # print('last_14_day筆數:', len(last_14_day))
# # print('類別:', last_14_day.__class__, ';', '陣列形狀:', last_14_day.shape, '陣列維度:', last_14_day.ndim)
# #
# # # 製作 samples
# #
# # # 改維度: 2D -> 3D
# # last_14_day_3D = np.reshape(last_14_day, (1, rolling_time, features))
# #
# # print('last_14_day_3D:', len(last_14_day_3D))
# # print('類別:', last_14_day_3D.__class__, ';', '陣列形狀:', last_14_day_3D.shape, '陣列維度:', last_14_day_3D.ndim, '\n')
# #
# # # 丟X_test_3D值進去訓練好的模型，預測 y_test
# # predicted_1Day = model.predict(last_14_day_3D)
# #
# # # to get the original scale
# # predicted_1Day = sc.inverse_transform(predicted_1Day)
# #
# # predicted_price_1Day = predicted_1Day.reshape(-1,1)[0:1, 0][0]
# # predicted_amount_1Day = predicted_1Day.reshape(-1,1)[1:2, 0][0]
# #
# # # numpy Array 取值 語法 array[row:row, col] －> 取值完後變list
# #
# # print('模型預測　未來1日　價格:', round(predicted_price_1Day,2))
# # print('模型預測　未來1日　量:', round(predicted_amount_1Day,2))
# #
# # # # 14. 歷史價格 vs 模型預測 y_train & y_test歷史價格
# #
# # # -------- 測試集資料  -------
# #
# # Predicted_future_list = [] # 測試集資料
# #
# # # 將 nan值 新增到 Predicted_future_list
# # for i in range(n):
# #     Predicted_future_list.append([np.nan])
# #
# # print('Predicted_future_list 資料筆數:', len(Predicted_future_list))
# # print('Predicted_future_list 類別:', Predicted_future_list.__class__)
# # print('查看 最後 rolling_time 筆 內容:\n', Predicted_future_list[-rolling_time: ]) # 用切片查看 最後 rolling_time 筆 資料內容
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # # 將 模型預測的 y_test 之7筆價格 新增到 Predicted_future_list
# # for i in predicted_price_y_test: # predicted_price_y_test是 模型預測的 y_test 價格
# #     # print(i)
# #     Predicted_future_list.append(list(i))
# #
# # print('Predicted_future_list 資料筆數:', len(Predicted_future_list))
# # print('Predicted_future_list 類別:', Predicted_future_list.__class__)
# # print('查看 最後 rolling_time 筆 內容:\n', Predicted_future_list[-rolling_time:]) # 用切片查看 最後 rolling_time 筆 資料內容
# #
# # # -------- 訓練集資料 -------
# #
# # # 讀取 所有原始資料 「price」欄位的值
# # dataset_all = dataset.iloc[:, 1:2].values  # dataset是原始資料
# #
# # print('dataset_all 筆數:', len(dataset_all))
# # print('類別:', dataset_all.__class__, ';', 'dataset_all 陣列形狀:',dataset_all.shape, 'dataset_all 陣列維度:', dataset_all.ndim)
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # # 訓練集資料正規化
# # dataset_all_scaled = sc.transform(dataset_all)
# # print('inputs_scaled:\n', dataset_all_scaled)
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # aX_test_list = []
# # for i in range(rolling_time, len(dataset)):
# #     aX_test_list.append(dataset_all_scaled[i-rolling_time:i, 0])
# # print('aX_test_list 筆數:', len(aX_test_list))
# # print('aX_test_list 類別:', aX_test_list.__class__)
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # # 訓練集list 轉 array
# # aX_test_2D = np.array(aX_test_list)
# # print('aX_test_2D 類別:', aX_test_2D.__class__, ';', '陣列形狀:', aX_test_2D.shape, ';', '陣列維度:', aX_test_2D.ndim)
# # print('筆數:', len(aX_test_2D))
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # # 訓練集2D 轉 3D
# # aX_test_3D = np.reshape(aX_test_2D, (aX_test_2D.shape[0], aX_test_2D.shape[1], 1))
# # print('aX_test_3D 類別:', aX_test_3D.__class__, ';', '陣列形狀:', aX_test_3D.shape, ';', '陣列維度:', aX_test_3D.ndim)
# # print('筆數:', len(aX_test_3D))
# #
# # # 將aX_test_3D值進去模型，預測 ay_test
# # predicted_train_price = model.predict(aX_test_3D)
# #
# # # to get the original scale
# # predicted_train_price = sc.inverse_transform(predicted_train_price)
# # print('ay_test 筆數:', len(predicted_train_price))
# #
# # Predicted_before_list = [] # 訓練集資料
# #
# # # 將 模型預測的 ay_test 之價格 新增到 Predicted_before_list
# # for i in predicted_train_price:
# #     Predicted_before_list.append(list(i))
# #
# # print('Predicted_before_list 資料筆數:', len(Predicted_before_list))
# # print('Predicted_before_list 類別:', Predicted_before_list.__class__)
# # print('查看 前 rolling_time 筆 :\n', Predicted_before_list[:rolling_time]) # 用切片查看 前 rolling_time 筆 資料內容
# # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', '\n')
# #
# # # 將 nan值 新增到 Predicted_before_list
# # for i in range(rolling_time):
# #     Predicted_before_list.append([np.nan])
# #
# # print('Predicted_before_list 資料筆數:', len(Predicted_before_list))
# # print('Predicted_before_list 類別:', Predicted_before_list.__class__)
# # print('查看 最後7筆 :\n', Predicted_before_list[-rolling_time:]) # 用切片查看 最後 rolling_time 筆 資料內容
# #
# # #  畫圖
# #
# # plt.plot(dataset_all, color = 'red', label = 'Real History Price')  # 紅線表示真實價格
# # plt.plot(Predicted_before_list, color = 'blue', label = 'Predicted history Price for train')  # 藍線表示預測歷史價格(訓練資料-驗證)
# # plt.plot(Predicted_future_list, color = 'green', label = 'Predicted history Price for test')  # 綠線表示預測歷史價格(測試資料)
# #
# # plt.title('Price forecast')
# # plt.xlabel('Day')
# # plt.ylabel('Avg Price')
# # plt.legend()
# # plt.show()
# #
# # # # calculate root mean squared error
# # # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# # # print('Train Score: %.2f RMSE' % (trainScore))
# # # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# # # print('Test Score: %.2f RMSE' % (testScore))
