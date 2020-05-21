
import requests
import time
import json
import MySQLdb

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/80.0.3987.132 Safari/537.36'}
url_head = 'https://data.coa.gov.tw/Service/OpenData/FromM/FarmTransData.aspx?$top=10000&$skip=0&CropCode=A1&StartDate='
ss = requests.session()

# Connect the database
# db = MySQLdb.connect(host='localhost', user='root', passwd='root', db='banana_database', port=3306, charset='utf8')

db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')

cursor = db.cursor()
db.autocommit(True)

for date in range(101, 109):
    url = url_head + str(date) + '.01.01&EndDate=' + str(date) + '.12.31'
    res = ss.get(url, headers=headers)

    for data_dict in json.loads(res.text):
        try:
            sql_str = f"INSERT INTO banana (transaction_date, crop_number, crop_name, market_number, market_name, " \
                      f"high_price, middle_price, low_price, avg_price, volume) VALUES	(\'{data_dict['交易日期']}\', " \
                      f"\'{data_dict['作物代號']}\', \'{data_dict['作物名稱']}\', \'{data_dict['市場代號']}\', " \
                      f"\'{data_dict['市場名稱']}\', \'{data_dict['上價']}\', {data_dict['中價']}," \
                      f" {data_dict['下價']}, {data_dict['平均價']}, {data_dict['交易量']}); "

            cursor.execute(sql_str)
            print('Done')
        except Exception as err:
            print(err.args)

    print('--------------------')
    time.sleep(3)

# url = url_head + '109.01.01&EndDate=109.04.30'
# res = ss.get(url, headers=headers)
#
# for data_dict in json.loads(res.text):
#     try:
#         sql_str = f"INSERT INTO adler_banana2 (transaction_date, crop_number, crop_name, market_number, market_name, " \
#                   f"high_price, middle_price, low_price, avg_price, volume) VALUES	(\'{data_dict['交易日期']}\', " \
#                   f"\'{data_dict['作物代號']}\', \'{data_dict['作物名稱']}\', \'{data_dict['市場代號']}\', " \
#                   f"\'{data_dict['市場名稱']}\', \'{data_dict['上價']}\', {data_dict['中價']}," \
#                   f" {data_dict['下價']}, {data_dict['平均價']}, {data_dict['交易量']}); "
#
#         cursor.execute(sql_str)
#         print('Done')
#     except Exception as err:
#         print(err.args)
#
# print('--------------------')
# time.sleep(3)
