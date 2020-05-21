SELECT 	*
FROM	trade_raws;

SELECT	t.trade_date, t.item_no, t.market_no, t.avg_price as 'price', t.trade_amount as 'amount'
FROM	trade_raws as t;