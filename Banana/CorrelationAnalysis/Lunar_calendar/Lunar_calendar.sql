select	*
from	date_map;

select distinct trade_date
from trade_raws;

select 	*
from	trade_raws;

select 	t.trade_date, t.item_no, t.avg_price, d.WDate, d.LDay
from	trade_raws as t
join	date_map as d
on		t.trade_date = d.WDate
where	t.item_no = 'A1';
