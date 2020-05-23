import MySQLdb
import datetime
import csv

# 開啟並建立csv檔案
csv_file = open('./job.csv', 'w+', newline='', encoding='utf8', errors='ignore')
csv_writer = csv.writer(csv_file)

# 建立表格欄位
table = ['year', 'price']
csv_writer.writerow(table)

# 連接Paul的資料庫
db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')

cursor = db.cursor()
db.autocommit(True)

date_list, price_list = [], []

sql_str = "select trade_date, avg_price, trade_amount from trade_raws_taipei;"
cursor.execute(sql_str)
data_rows = cursor.fetchall()

for row in data_rows:
    date_list.append(row[0])
    price_list.append(float(row[1]))

for i in range(len(date_list)):
    row = [date_list[i], (price_list[i])]
    csv_writer.writerow(row)

csv_file.close()
