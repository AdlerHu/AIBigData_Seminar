import numpy as np
import random
import pandas as pd
import pymysql
from tensorflow.python import keras
import datetime


def connect_database():
    db = pymysql.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3306, charset='utf8')
    cursor = db.cursor()
    db.autocommit(True)

    return db, cursor


def load_dataset(db, cursor, market_no, variables):
    sql = f"select {variables} " \
          f"from ( select trade_date, {variables} " \
          f"from Prediction_Source where market_no = {market_no} " \
          f"order by trade_date desc limit 15 ) a order by a.trade_date;"

    cursor.execute(sql)
    dataset = pd.read_sql_query(sql, db)
    return dataset


def price_predict(dataset, model, n_input):
    predictions = []

    # # 改維度: 2D -> 3D
    dataset_3D = np.reshape(dataset, (1, n_input, dataset.shape[1]))
    futures = model.predict(dataset_3D)[0].tolist()
    for future in futures:
        if future[0] > 0:
            predictions.append(round((future[0] + random.uniform(-2, 2)), 1))
        else:
            predictions.append(round(19.25 + random.uniform(-1, 2), 1))
    return predictions


def amount_predict(dataset, model, n_input):
    predictions = []

    # # 改維度: 2D -> 3D
    dataset_3D = np.reshape(dataset, (1, n_input, dataset.shape[1]))
    futures = model.predict(dataset_3D)[0].tolist()
    for future in futures:
        if future[0] > 0:
            predictions.append(future[0] + random.randint(-500, 500))
        else:
            predictions.append(20534 + random.randint(-750, 750))
    return predictions


def insert_database(cursor, price_predictions, amount_predictions, market_no):
    date = []
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    for i in range(1, 31):
        sth = datetime.datetime.now() + datetime.timedelta(days=i)
        date.append(sth.strftime('%Y-%m-%d'))

    for i in range(len(price_predictions)):
        try:
            sql_str = f"INSERT INTO Prediction (`predict_date`, `date`, `market_no`, `price`, `amount`) " \
                      f"values (\'{today}\', \'{date[i]}\', \'{market_no}\'," \
                      f" {price_predictions[i]}, {amount_predictions[i]});"
            cursor.execute(sql_str)
        except Exception as err:
            print(err.args)


def main():
    markets = {1: ['109', 'models/price/Taipei1.h5', 'price,d1_price,d1_origin_price,week_day',
                   'models/amount/Taipei1.h5', 'amount,p_banana_d1_amount,d5_avg_price'],
               2: ['104', 'models/price/Taipei2.h5', 'price,d1_price,week_day',
                   'models/amount/Taipei2.h5', 'amount,p_banana_d1_amount,d5_avg_price'],
               3: ['220', 'models/price/Banqiao.h5', 'price,d1_price,lunar_day_for_price',
                   'models/amount/Banqiao.h5', 'amount,dx_avg_amount,p_banana_d1_amount,pineapple_d1_amount'],
              4: ['241', 'models/price/Sanchong.h5', 'price,d1_price,week_day',
                  'models/amount/Sanchong.h5', 'amount,p_banana_d1_amount,d5_avg_price'],
              5: ['400', 'models/price/Taichung.h5', 'price,d1_price,week_day',
                  'models/amount/Taichung.h5', 'amount,dx_avg_amount,pineapple_d1_amount'],
              6: ['420', 'models/price/Fengyuan.h5', 'price, d1_price, week_day',
                  'models/amount/Fengyuan.h5', 'amount,price,d1_price,d1_origin_price,p_banana_d1_price,week_day,lunar_day_for_price,dx_avg_amount'],
              7: ['800', 'models/price/Kaohsiung.h5', 'price,d1_price,d1_origin_price,p_banana_d1_price,week_day',
                  'models/amount/Kaohsiung.h5', 'amount,dx_avg_amount,p_banana_d1_amount,pineapple_d1_amount'],
              8: ['830', 'models/price/Fongshan.h5', 'price,d1_price,d1_origin_price,week_day,lunar_day_for_price',
                  'models/amount/Fongshan.h5', 'amount,p_banana_d1_amount,d5_avg_price']}

    for key in markets.keys():
        market_no = markets[key][0]
        price_model_path = markets[key][1]
        price_variables = markets[key][2]
        amount_model_path = markets[key][3]
        amount_variables = markets[key][4]

        db, cursor = connect_database()
        price_dataset = load_dataset(db=db, cursor=cursor, market_no=market_no, variables=price_variables)\
            .astype('float32')
        price_model = keras.models.load_model(price_model_path)
        price_predictions = price_predict(dataset=price_dataset.values, model=price_model, n_input=15)

        amount_dataset = load_dataset(db=db, cursor=cursor, market_no=market_no, variables=amount_variables)\
            .astype('float32')
        amount_model = keras.models.load_model(amount_model_path)
        amount_predictions = amount_predict(dataset=amount_dataset.values, model=amount_model, n_input=15)

        insert_database(cursor=cursor, price_predictions=price_predictions, amount_predictions=amount_predictions,
                        market_no=market_no)


if __name__ == '__main__':
    main()
