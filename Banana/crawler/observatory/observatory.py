import requests
import pandas as pd
import MySQLdb

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/80.0.3987.132 Safari/537.36'}
city = ['南投縣', '嘉義縣', '高雄市', '屏東縣']

# 連接Paul的資料庫
db = MySQLdb.connect(host=host, user='dbuser', passwd=passwd, db='fruveg', port=3307, charset='utf8')

cursor = db.cursor()
db.autocommit(True)

ss = requests.session()
url = 'http://e-service.cwb.gov.tw/wdps/obs/state.htm'

res = ss.get(url, headers=headers)
res.encoding = 'big5'
df = pd.read_html(res.text)[0]

df2 = df.iloc[2:610]

station_code_list = list(df2[0])
station_name_list = list(df2[1])
city_list = list(df2[5])
start_date_list = list(df2[7])

date_list = []

for date in start_date_list:
    date_list.append(date.replace('/', '-'))

for i in range(len(city_list)):
    if city_list[i] in city:
        try:
            sql_str = f"INSERT INTO adler_observatory (`code`, `name`, `city`, `start_date`) " \
                      f"VALUES	(\'{station_code_list[i]}\', \'{station_name_list[i]}\', " \
                      f"\'{city_list[i]}\', \'{start_date_list[i]}\');"
            cursor.execute(sql_str)
            print('Done')
        except Exception as err:
            print(err.args)
            
db.close()
cursor.close()
