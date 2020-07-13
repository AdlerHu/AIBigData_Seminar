import requests
import pandas as pd
import MySQLdb
import datetime
import time


# 算出起始日期跟截止日期之間有幾天
def diff_days(begin_date, end_date):

    begin_year = int(begin_date.split('-')[0])
    begin_month = int(begin_date.split('-')[1])
    begin_day = int(begin_date.split('-')[2])
    end_year = int(end_date.split('-')[0])
    end_month = int(end_date.split('-')[1])
    end_day = int(end_date.split('-')[2])

    end_day = datetime.date(end_year, end_month, end_day)
    begin_day = datetime.date(begin_year, begin_month, begin_day)
    days = end_day - begin_day

    return days.days


# 連接資料庫
def connect_database():

    # 連接我的資料庫
    db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='root', db='banana_database', port=3306, charset='utf8')
    # 連接Paul的資料庫
    # db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')

    cursor = db.cursor()
    db.autocommit(True)

    return cursor


# 寫入資料庫
def insert_into_database(cursor, station, data, date):

    # 氣壓
    pressure_list = data['Press']['測站氣壓(hPa)']['StnPres']
    # 氣溫
    temperature_list = data['temperature']['氣溫(℃)']['Temperature']
    # 相對濕度
    humidity_list = data['RH']['相對溼度(%)']['RH']
    # 風速
    wind_speed_list = data['WD/WS']['風速(m/s)']['WS']
    # 風向
    wind_direction_list = data['WD/WS']['風向(360degree)']['WD']
    # 降水量
    precipitation_list = data['Precp']['降水量(mm)']['Precp']

    for i in range(len(precipitation_list)):
        try:
            sql_str = f"INSERT INTO Adler_weather (`station`, `date`, `hour`, `pressure`, " \
                      f"`temperature`, `humidity`, `wind_speed`, `wind_direction`, `precipitation`) " \
                      f"VALUES ('{station}',\'{date}\','{i + 1}', '{pressure_list[i]}', '{temperature_list[i]}'," \
                      f" '{humidity_list[i]}', '{wind_speed_list[i]}', '{wind_direction_list[i]}', " \
                      f"'{precipitation_list[i]}');"
            cursor.execute(sql_str)
            print('Done')
        except Exception as err:
            print(err.args)


# 從資料庫得到觀測站代號、起始日期
def get_station_information(cursor):

    station_code_list, date_list, start_date_list = [], [], []

    sql_str = f"SELECT `code`, `start_date` FROM Adler_observatory; "
    cursor.execute(sql_str)
    data_rows = cursor.fetchall()

    for row in data_rows:
        station_code_list.append(row[0])
        date_list.append(str(row[1]))

    # 觀測站起始日期如果比20120101早,就設定從20120101開始爬
    for date in date_list:
        d = int(date.replace('-', ''))
        if d < 20120101:
            start_date_list.append('2012-01-01')
        else:
            start_date_list.append(date)

    return station_code_list, start_date_list


def main(end_date):

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/80.0.3987.132 Safari/537.36'}
    ss = requests.session()
    cursor = connect_database()

    station_code_list, start_date_list = get_station_information(cursor=cursor)

    for i in range(len(station_code_list)):

        station = station_code_list[i]
        start_date = start_date_list[i]

        days = diff_days(begin_date=start_date, end_date=end_date)

        for day in range(days):
            dt = datetime.datetime.strptime(str(start_date), "%Y-%m-%d")
            date = (dt + datetime.timedelta(days=day)).strftime("%Y-%m-%d")

            url = 'https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=' \
                  + station + '&stname=%25E9%25B3%25B3%25E5%25B1%25B1&datepicker=' + date

            res = ss.get(url, headers=headers)
            data = pd.read_html(res.text)[1]

            insert_into_database(cursor=cursor, station=station, data=data, date=date)

            time.sleep(1)

    cursor.close()
            
            
if __name__ == '__main__':
    '''
    end: 終止日期, 格式為'YYYY-MM-DD'不補零
    '''

    end = '2019-12-31'
    main(end_date=end)
