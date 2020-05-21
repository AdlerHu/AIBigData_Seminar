library(RMySQL)
library(DBI)

dbconn <- dbConnect(RMySQL::MySQL(),host="127.0.0.1",user="dbuser",password="20200428",dbname="fruveg",port=3307)
dbGetQuery(dbconn, "SET NAMES 'big5'") # 解決繁體中文亂碼問題 

sql = "SELECT	t.trade_date, t.item_no, t.market_no, t.avg_price as 'price', t.trade_amount as 'amount'
FROM	trade_raws as t;"
dataf=dbGetQuery(dbconn,sql)

x=as.vector(t(dataf$price))
y=as.vector(t(dataf$amount))


#皮爾森相關係數Pearson Correlation
cor(x,y)
cor.test(x,y)
plot(x, y)
abline(lm(y~x),col="red")
legend("topleft", legend = c("r=-0.1547535 ","p-value=2.2e-16"))
