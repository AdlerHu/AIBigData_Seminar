select 	* 
from	Adler_gdp;

select	*
from	trade_raws;


Select 	annual, season, gdp,egr, round(avg(avg_price), 2) as a_pric, round(avg(trade_amount), 0) as a_amount
From (	Select annual, season, gdp,egr, avg_price, trade_amount
		From Adler_gdp a
		Left Join trade_raws_fongshan b on b.trade_date between a.start_date and a.end_date) a 
Group by annual, season;


Select 	annual, season, gdp,egr, round(avg(avg_price), 2) as a_pric, round(avg(trade_amount), 0) as a_amount
From (	Select annual, season, gdp,egr, avg_price, trade_amount
		From Adler_gdp a
		Left Join trade_raws_fongshan b on b.trade_date between a.start_date and a.end_date) a 
Group by annual, season;
