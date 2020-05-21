USE fruveg;

Select a.num, a.trade_date, (a.avg_price-b.avg_price) as d_price, 
       (a.trade_amount-b.trade_amount) as d_amount
From
(Select num, trade_date, avg_price, trade_amount
 From trade_raws_temp) a
Left Join 
(Select num, trade_date, avg_price, trade_amount
 From trade_raws_temp) b on a.num=b.num+1
Where b.num is not NULL
Order by a.num;

