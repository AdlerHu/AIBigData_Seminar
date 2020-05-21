import MySQLdb
import datetime
import os
import csv
import pprint
"""建立資料夾路徑"""
resource_path = './file'
if not os.path.exists(resource_path):
    os.mkdir(resource_path)
"""開啟並建立csv檔案"""
csv_file = open('./file/job.csv', 'w+', newline='', encoding='Big5', errors='ignore')
csv_writer = csv.writer(csv_file)

"""建立表格欄位"""

table = ['年度', '均價']
csv_writer.writerow(table)


# db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='root', db='banana_database', port=3306, charset='utf8')
# 連接Paul的資料庫
db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')

cursor = db.cursor()
db.autocommit(True)

x, y = [], []

sql_str = "select * from trade_raws where item_no='A1';"
cursor.execute(sql_str)
data_rows = cursor.fetchall()
# pprint.pprint(data_rows)
for row in data_rows:
    # gdp_list.append(row)
    x.append(row[0])
    y.append(row[8])

# print(x)
# print(y)


for i in range(len(x)):
    row = [x[i], (y[i])]
    csv_writer.writerow(row)

csv_file.close()
