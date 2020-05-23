import MySQLdb
import datetime
import os
import csv
import pprint
import pandas as pd  # pandas庫
import numpy as np  # numpy庫
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # acf和pacf展示庫
from statsmodels.tsa.stattools import adfuller  # adf檢驗庫
from statsmodels.stats.diagnostic import acorr_ljungbox  # 隨機性檢驗庫
from statsmodels.tsa.arima_model import ARMA  # ARMA庫
import matplotlib.pyplot as plt  # matplotlib圖形展示庫
import prettytable  # 匯入表格庫
# 由於pandas判斷週期性時會出現warning，這裡忽略提示
import warnings


# 多次用到的表格
def pre_table(table_name, table_rows):
    """
    :param table_name: 表格名稱，字串列表
    :param table_rows: 表格內容，巢狀列表
    :return: 展示表格物件
    """
    table = prettytable.PrettyTable()  # 建立表格例項
    table.field_names = table_name  # 定義表格列名
    for i in table_rows:  # 迴圈讀多條資料
        table.add_row(i)  # 增加資料
    return table


# 穩定性（ADF）檢驗
def adf_val(ts, ts_title, acf_title, pacf_title):
    """
    :param ts: 時間序列資料，Series型別
    :param ts_title: 時間序列圖的標題名稱，字串
    :param acf_title: acf圖的標題名稱，字串
    :param pacf_title: pacf圖的標題名稱，字串
    :return: adf值、adf的p值、三種狀態的檢驗值
    """
    plt.figure()
    plt.plot(ts)  # 時間序列圖
    plt.title(ts_title)  # 時間序列標題
    plt.show()
    plot_acf(ts, lags=5, title=acf_title).show()  # 自相關檢測
    plot_pacf(ts, lags=5, title=pacf_title).show()  # 偏相關檢測
    adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(ts)  # 穩定性（ADF）檢驗
    table_name = ['adf', 'pvalue', 'usedlag', 'nobs', 'critical_values', 'icbest']  # 表格列名列表
    table_rows = [[adf, pvalue, usedlag, nobs, critical_values, icbest]]  # 表格行資料，巢狀列表
    adf_table = pre_table(table_name, table_rows)  # 獲得平穩性展示表格物件
    print ('stochastic score')  # 列印標題
    print (adf_table)  # 列印展示表格
    return adf, pvalue, critical_values,  # 返回adf值、adf的p值、三種狀態的檢驗值


# 白噪聲（隨機性）檢驗
def acorr_val(ts):
    """
    :param ts: 時間序列資料，Series型別
    :return: 白噪聲檢驗的P值和展示資料表格物件
    """
    lbvalue, pvalue = acorr_ljungbox(ts, lags=1)  # 白噪聲檢驗結果
    table_name = ['lbvalue', 'pvalue']  # 表格列名列表
    table_rows = [[lbvalue, pvalue]]  # 表格行資料，巢狀列表
    acorr_ljungbox_table = pre_table(table_name, table_rows)  # 獲得白噪聲檢驗展示表格物件
    print ('stationarity score')  # 列印標題
    print (acorr_ljungbox_table)  # 列印展示表格
    return pvalue  # 返回白噪聲檢驗的P值和展示資料表格物件


# 資料平穩處理
def get_best_log(ts, max_log=2, rule1=True, rule2=True):
    """
    :param ts: 時間序列資料，Series型別
    :param max_log: 最大log處理的次數，int型
    :param rule1: rule1規則布林值，布林型
    :param rule2: rule2規則布林值，布林型
    :return: 達到平穩處理的最佳次數值和處理後的時間序列
    """
    if rule1 and rule2:  # 如果兩個規則同時滿足
        return 0, ts  # 直接返回0和原始時間序列資料
    else:  # 只要有一個規則不滿足
        for i in range(1, max_log):  # 迴圈做log處理
            ts = np.log(ts)  # log處理
            adf, pvalue1, usedlag, nobs, critical_values, icbest = adfuller(ts)  # 穩定性（ADF）檢驗
            lbvalue, pvalue2 = acorr_ljungbox(ts, lags=1)  # 白噪聲（隨機性）檢驗
            rule_1 = (adf < critical_values['1%'] and adf < critical_values['5%'] and adf < critical_values[
                '10%'] and pvalue1 < 0.01)  # 穩定性（ADF）檢驗規則
            rule_2 = (pvalue2 < 0.05)  # 白噪聲（隨機性）規則
            rule_3 = (i < 5)
            if rule_1 and rule_2 and rule_3:  # 如果同時滿足條件
                print ('The best log n is: {0}'.format(i))  # 列印輸出最佳次數
                return i, ts  # 返回最佳次數和處理後的時間序列


# 還原經過平穩處理的資料
def recover_log(ts, log_n):
    """
    :param ts: 經過log方法平穩處理的時間序列，Series型別
    :param log_n: log方法處理的次數，int型
    :return: 還原後的時間序列
    """
    for i in range(1, log_n + 1):  # 迴圈多次
        ts = np.exp(ts)  # log方法還原
    return ts  # 返回時間序列


warnings.filterwarnings('ignore')

# 讀取 CSV 檔案內容
row = pd.read_csv('./job.csv', encoding='utf8', index_col='year')

ts_data = row['price'].astype('float')

# 原始資料檢驗
# 穩定性檢驗
adf, pvalue1, critical_values = adf_val(ts_data, 'raw time series', 'raw acf', 'raw pacf')
# 白噪聲檢驗
pvalue2 = acorr_val(ts_data)


# 對時間序列做穩定性處理
# 穩定性檢驗規則
rule1 = (adf < critical_values['1%'] and adf < critical_values['5%'] and adf < critical_values[
    '10%'] and pvalue1 < 0.01)
# 白噪聲檢驗的規則
rule2 = (pvalue2[0,] < 0.05)
# 使用log進行穩定性處理
log_n, ts_data = get_best_log(ts_data, rule1=rule1, rule2=rule2)
