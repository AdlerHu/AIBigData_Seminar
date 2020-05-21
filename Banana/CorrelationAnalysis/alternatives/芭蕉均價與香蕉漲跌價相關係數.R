library(RMySQL)
library(DBI)

dbconn <- dbConnect(RMySQL::MySQL(),host="127.0.0.1",user="dbuser",password="20200428",dbname="fruveg",port=3307)
dbGetQuery(dbconn, "SET NAMES 'big5'") # 解決繁體中文亂碼問題 

sql = "select 	a.*, b.avg_price as 'ba_avg_price', b.trade_date
from	(	Select c.num, c.trade_date, c.avg_price, c.trade_amount, (c.avg_price - d.avg_price) as d_avg_price, (c.trade_amount - d.trade_amount) as d_trade_amount
			From trade_raws_taipei c
			Left Join trade_raws_taipei d on c.num=d.num+1
			Where d.num is not null) as a
join	(	select	*
			from	trade_raws as t
			where	item_no = 'A3' and market_no = '104') as b
on a.trade_date = b.trade_date;"
dataf=dbGetQuery(dbconn,sql)

x=as.vector(t(dataf$ba_avg_price))
y=as.vector(t(dataf$trade_amount))


#皮爾森相關係數Pearson Correlation
cor(x,y)
cor.test(x,y)
plot(x, y)
abline(lm(y~x),col="red")
legend("topleft", legend = c("r=-0.3071997","p-value=.2e-16"))