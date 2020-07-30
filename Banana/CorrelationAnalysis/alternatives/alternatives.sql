SELECT 	*
FROM	trade_raws;

SELECT 	distinct item_no, item_name
FROM	trade_raws;

# 價-價, 量-量
SELECT	a.trade_date, a.item_no, a.market_no,a.avg_price as 'banana_avg_price' ,a.trade_amount as 'banana_amount', b.trade_date, b.item_no, b.market_no,b.avg_price as 'alternative_avg_price', b.trade_amount as 'alternative_amount'
FROM	trade_raws as a
JOIN	trade_raws as b
ON		(a.market_no = b.market_no) AND (a.trade_date = b.trade_date)
WHERE	(a.item_no = 'A1') AND (b.item_no = 'A2');

# 價差、量差
select x.d_avg_price, x.d_trade_amount, a.avg_price, a.trade_amount
from (  Select c.item_no, c.trade_date, (c.avg_price - d.avg_price) as d_avg_price, (c.trade_amount - d.trade_amount) as d_trade_amount
		From trade_raws_taipei c
		Left Join trade_raws_taipei d on c.num=d.num+1
		Where d.num is not null) x
join	trade_raws as a
on		(x.trade_date = a.trade_date) and (a.market_no = '109') and (a.item_no = 'A2');

# -------------------------------------------------------------------------------------------------

Select x.*,
       -- 是星期一給1, 其他給 0
    case when y.wday_name is null then 0 else 1 end as is_monday        
From 
(-- 2. 建立量價, 及量價差table
 Select c.num, c.trade_date, c.avg_price, c.trade_amount, (c.avg_price - d.avg_price) as d_avg_price, (c.trade_amount - d.trade_amount) as d_trade_amount
 From trade_raws_taichung c
 Left Join trade_raws_taichung d on c.num=d.num+1
 Where d.num is not null) x
Left Join 
(-- 1.先找出星期一的交易日
 Select a.num, b.wday_name
 From trade_raws_taichung a
 Left Join date_map b on a.trade_date=b.wdate
 Where b.wday_name='一') y on x.num=y.num;


