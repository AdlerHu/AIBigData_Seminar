select 	a.*, b.avg_price as 'ba_avg_price', b.trade_date
from	(	Select c.num, c.trade_date, c.avg_price, c.trade_amount, (c.avg_price - d.avg_price) as d_avg_price, (c.trade_amount - d.trade_amount) as d_trade_amount
			From trade_raws_taipei c
			Left Join trade_raws_taipei d on c.num=d.num+1
			Where d.num is not null) as a
join	(	select	*
			from	trade_raws as t
			where	item_no = 'A2' and market_no = '104') as b
on a.trade_date = b.trade_date;
