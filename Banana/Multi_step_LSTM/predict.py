import numpy as np
import random
import pandas as pd
# import MySQLdb
from tensorflow import keras
import datetime
import pymysql


def connect_database():
#     db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')
    db = pymysql.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3306, charset='utf8')
    cursor = db.cursor()
    db.autocommit(True)

    return db, cursor


def load_dataset(db, cursor):
    sql = '''select a.price, a.d1_price, a.d1_origin_price, a.week_day from ( 
            select trade_date, price, d1_price, d1_origin_price, week_day from Prediction_Source 
                where market_no = '109'
                order by trade_date desc
                limit 15 ) a
    order by a.trade_date;'''

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


def insert_database(db, cursor, predictions):
    date = []
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    for i in range(1, 31):
        sth = datetime.datetime.now() + datetime.timedelta(days=i)
        date.append(sth.strftime('%Y-%m-%d'))

    for i in range(len(predictions)):
        try:
            sql_str = f"INSERT INTO Taipei1_price_prediction (`predict_date`, `date`, `price`) " \
                      f"values (\'{today}\', \'{date[i]}\', {predictions[i]});"
            cursor.execute(sql_str)
            print('Done')
        except Exception as err:
            print(err.args)


def main():
    db, cursor = connect_database()
    dataset = load_dataset(db, cursor)
    model = keras.models.load_model('LSTM_taipei1(15to30).h5')
    predictions = predict(dataset=dataset.values, model=model, n_input=15)
    insert_database(db, cursor, predictions)


if __name__ == '__main__':
    main()
