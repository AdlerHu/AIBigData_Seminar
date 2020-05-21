library(RMySQL)
library(DBI)

dbconn <- dbConnect(RMySQL::MySQL(),host="127.0.0.1",user="dbuser",password="20200428",dbname="fruveg",port=3307)
dbGetQuery(dbconn, "SET NAMES 'big5'") # 解決繁體中文亂碼問題 

sql = "Select x.*,
       -- 是星期一給1, 其他給 0
    case when y.wday_name is null then 0 else 1 end as is_thursday        
From 
(-- 2. 建立量價, 及量價差table
 Select c.num, c.trade_date, c.avg_price, c.trade_amount,
        (c.avg_price - d.avg_price) as d_avg_price, 
        (c.trade_amount - d.trade_amount) as d_trade_amount
 From trade_raws_taichung c
 Left Join trade_raws_taichung d on c.num=d.num+1
 Where d.num is not null) x
Left Join 
(-- 1.先找出星期一的交易日
 Select a.num, b.wday_name
 From trade_raws_taichung a
 Left Join date_map b on a.trade_date=b.wdate
 Where b.wday_name='二') y on x.num=y.num;"
dataf=dbGetQuery(dbconn,sql)

x=as.vector(t(dataf$d_trade_amount))
y=as.vector(t(dataf$is_thursday))


#皮爾森相關係數Pearson Correlation
cor(x,y)
cor.test(x,y)
plot(x, y)
abline(lm(y~x),col="red")
legend("topleft", legend = c("r=-0.1467897","p-value=5.234e-13"))
