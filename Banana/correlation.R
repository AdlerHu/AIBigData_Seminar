library(RMySQL)
library(DBI)

dbconn <- dbConnect(RMySQL::MySQL(),host="127.0.0.1",user="dbuser",password="20200428",dbname="fruveg",port=3307)
dbGetQuery(dbconn, "SET NAMES 'big5'") # 解決繁體中文亂碼問題 

sql = "Select a.num, a.trade_date, (a.avg_price-b.avg_price) as d_price, 
       (a.trade_amount-b.trade_amount) as d_amount
From
(Select num, trade_date, avg_price, trade_amount
 From trade_raws_temp) a
Left Join 
(Select num, trade_date, avg_price, trade_amount
 From trade_raws_temp) b on a.num=b.num+1
Where b.num is not NULL
Order by a.num;"
dataf=dbGetQuery(dbconn,sql)

x=as.vector(t(dataf$d_price))
y=as.vector(t(dataf$d_amount))


#皮爾森相關係數Pearson Correlation
cor(x,y)
cor.test(x,y)
plot(x, y)
abline(lm(y~x),col="red")
legend("topleft", legend = c("r=0.01770002","p-value=2.2e-16"))

