# 前一日產地價、前一日價量、前一日芭蕉價量
SELECT t.trade_date, t.item_no, t.avg_price, t.trade_amount, o.trade_date, o.avg_price AS 'origin_price', x.trade_date, 
		x.avg_price AS 'yesterday_price', x.trade_amount AS 'yesterday_amount', a.trade_date, a.avg_price AS 'alternative_price', a.trade_amount AS 'alternative_amount'
FROM trade_raws_taipei1 t
JOIN adler_origin_avg_price_day o
ON t.num = o.num + 1
JOIN trade_raws_taipei1 x
ON t.num = x.num + 1
JOIN adler_pbanana_taipei1 a
ON t.num = a.num + 1
;
