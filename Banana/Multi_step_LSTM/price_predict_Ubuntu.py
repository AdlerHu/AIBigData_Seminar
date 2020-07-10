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


def predict(dataset, model, n_input):
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


def insert_database(db, cursor, predictions, market_no):
    date = []
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    for i in range(1, 31):
        sth = datetime.datetime.now() + datetime.timedelta(days=i)
        date.append(sth.strftime('%Y-%m-%d'))

    for i in range(len(predictions)):
        try:
            sql_str = f"INSERT INTO price_prediction (`predict_date`, `date`, `market_no`, `price`) " \
                      f"values (\'{today}\', \'{date[i]}\', \'{market_no}\', {predictions[i]});"
            cursor.execute(sql_str)
        except Exception as err:
            print(err.args)


def main():
    markets = {1: ['price_prediction', '109', 'models/price/Taipei1.h5', 'price,d1_price,d1_origin_price,week_day']}

    for key in markets.keys():
        market_no = markets[key][1]
        model_path = markets[key][2]
        variables = markets[key][3]

        db, cursor = connect_database()
        dataset = load_dataset(db=db, cursor=cursor, market_no=market_no, variables=variables).astype('float32')
        model = keras.models.load_model(model_path)
        predictions = predict(dataset=dataset.values, model=model, n_input=15)
        insert_database(db=db, cursor=cursor, predictions=predictions, market_no=market_no)


if __name__ == '__main__':
    main()
