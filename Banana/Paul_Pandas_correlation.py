import MySQLdb
import pandas as pd

db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307,
                     charset='utf8')
cursor = db.cursor()

sql = """Select gdp,egr, round(avg(avg_price), 2) as '3個月均價', round(avg(trade_amount), 0) as '平均交易量'
From (	Select annual, season, gdp,egr, avg_price, trade_amount
		From Adler_gdp a Left Join trade_raws_fongshan b on b.trade_date 
		between a.start_date and a.end_date) a Group by annual, season;
"""

cursor.execute(sql)
df = pd.read_sql_query(sql, db)
pearsoncorr = df.corr(method='pearson')
print(pearsoncorr)
db.close()
