library(RMySQL)
library(DBI)

dbconn <- dbConnect(RMySQL::MySQL(),host="127.0.0.1",user="dbuser",password="20200428",dbname="fruveg",port=3307)
dbGetQuery(dbconn, "SET NAMES 'big5'") # 解決繁體中文亂碼問題 

sql = "SELECT	a.trade_date, a.item_no, a.market_no,a.avg_price as 'banana_avg_price' ,
a.trade_amount as 'banana_amount', b.trade_date, 
b.item_no, b.market_no,b.avg_price as 'alternative_avg_price', b.trade_amount as 'alternative_amount'
FROM	trade_raws as a
JOIN	trade_raws as b
ON		(a.market_no = b.market_no) AND (a.trade_date = b.trade_date)
WHERE	(a.item_no = 'A1') AND (b.item_no = 'A2');"
dataf=dbGetQuery(dbconn,sql)

x=as.vector(t(dataf$alternative_amount))
y=as.vector(t(dataf$banana_amount))


#皮爾森相關係數Pearson Correlation
cor(x,y)
cor.test(x,y)

plot(x, y)
abline(lm(y~x),col="red")
legend("topleft", legend = c("r=0.4952393","p-value=2.2e-16"))


# -----------------------------------------------------------------------------------

library(RMySQL)
library(DBI)

dbconn <- dbConnect(RMySQL::MySQL(),host="127.0.0.1",user="dbuser",password="20200428",dbname="fruveg",port=3307)
dbGetQuery(dbconn, "SET NAMES 'big5'") # 解決繁體中文亂碼問題 

sql = "select x.*, a.item_no, a.market_no
from (  Select c.num,c.item_no, c.market_no, c.trade_date, c.avg_price, c.trade_amount, (c.avg_price - d.avg_price) as d_avg_price, (c.trade_amount - d.trade_amount) as d_trade_amount
		From trade_raws_taichung c
		Left Join trade_raws_taichung d on c.num=d.num+1
		Where d.num is not null) x
join	trade_raws as a
on		(x.trade_date = a.trade_date) and (a.market_no = '400') and (a.item_no = 'A2');"
dataf=dbGetQuery(dbconn,sql)

x=as.vector(t(dataf$banana_amount))
y=as.vector(t(dataf$ba_amount))


#皮爾森相關係數Pearson Correlation
cor(x,y)
cor.test(x,y)
plot(x, y)
abline(lm(y~x),col="red")
legend("topleft", legend = c("r=0.1781265","p-value=2.2e-16"))
