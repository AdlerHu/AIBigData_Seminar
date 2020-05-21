import requests
import pandas as pd
import MySQLdb

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/80.0.3987.132 Safari/537.36'}

# 連接我的資料庫
db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='root', db='banana_database', port=3306, charset='utf8')
# 連接Paul的資料庫
# db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')

cursor = db.cursor()
db.autocommit(True)

data = {'queryStartDate': '2012/01/01', 'queryEndDate': '2019/12/31'}

url = 'https://www.taifex.com.tw/cht/3/dailyFXRate'
ss = requests.session()

resp = ss.post(url, headers=headers, data=data)

df = pd.read_html(resp.text)[2]
date_list = df['日期']
rate_list = df['美元／新台幣']

for date in date_list:
    date.replace('/', '-')


for i in range(len(rate_list)):
    try:
        sql_str = f"INSERT INTO Adler_rate (`date`, `rate`) VALUES (\'{date_list[i]}\', {rate_list[i]});"
        cursor.execute(sql_str)
        print('Done')
    except Exception as err:
        print(err.args)
