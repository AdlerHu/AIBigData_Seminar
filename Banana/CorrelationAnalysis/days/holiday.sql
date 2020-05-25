select 	WDate, d.Holiday_Name, date_add(d.WDate, interval -1 day) as '1 day before', date_add(d.WDate, interval -2 day) as '2 day before', date_add(d.WDate, interval -3 day) as '3 day before', date_add(d.WDate, interval -4 day) as '4 day before',
date_add(d.WDate, interval -5 day) as '5 day before', date_add(d.WDate, interval -6 day) as '6 day before', date_add(d.WDate, interval -7 day) as '7 day before'
from	date_map as d
where d.Holiday_Name = '清明' or d.Holiday_Name = '端午' or d.Holiday_Name = '中元' or d.Holiday_Name = '中秋' or d.Holiday_Name = '除夕';

select distinct Holiday_Name from date_map;

select * from trade_raws_taipei;

select * from date_map;

select 	t.*, 
	case when t.trade_date = x.one_day_before then 1 else 0 end as one_day,   
	case when t.trade_date = x.two_day_before then 2 else 0 end as two_day,
	case when t.trade_date = x.three_day_before then 3 else 0 end as three_day,
	case when t.trade_date = x.four_day_before then 4 else 0 end as four_day,
	case when t.trade_date = x.five_day_before then 5 else 0 end as five_day,    
	case when t.trade_date = x.six_day_before then 6 else 0 end as six_day,    
	case when t.trade_date = x.seven_day_before then 7 else 0 end as seven_day
    
from	trade_raws_taipei as t
join	( select WDate, date_add(d.WDate, interval -1 day) as 'one_day_before', date_add(d.WDate, interval -2 day) as 'two_day_before', date_add(d.WDate, interval -3 day) as 'three_day_before', 
			date_add(d.WDate, interval -4 day) as 'four_day_before', date_add(d.WDate, interval -5 day) as 'five_day_before', 
            date_add(d.WDate, interval -6 day) as 'six_day_before', date_add(d.WDate, interval -7 day) as 'seven_day_before' 
			from	date_map as d
			where d.Holiday_Name = '清明' or d.Holiday_Name = '端午' or d.Holiday_Name = '中元' or d.Holiday_Name = '中秋' or d.Holiday_Name = '除夕') as x
on	t.trade_date = x.one_day_before or t.trade_date = x.two_day_before or t.trade_date = x.three_day_before or 
t.trade_date = x.four_day_before or t.trade_date = x.five_day_before or t.trade_date = x.six_day_before or t.trade_date = x.seven_day_before;


select WDate, date_add(d.WDate, interval -1 day) as 'one_day_before', date_add(d.WDate, interval -2 day) as 'two_day_before', date_add(d.WDate, interval -3 day) as 'three_day_before', 
			date_add(d.WDate, interval -4 day) as 'four_day_before', date_add(d.WDate, interval -5 day) as 'five_day_before', 
            date_add(d.WDate, interval -6 day) as 'six_day_before', date_add(d.WDate, interval -7 day) as 'seven_day_before', d.Holiday_Name 
			from	date_map as d
			where (d.Holiday_Name = '清明' or d.Holiday_Name = '端午' or d.Holiday_Name = '中元' or d.Holiday_Name = '中秋' or d.Holiday_Name = '除夕') and (d.WDate between '2012-01-01' and '2020-05-15');
            
            
            
            
            
            
            
            
            
            
select 	t.*, 
	(case 
		when t.trade_date = x.one_day_before then 1
		when t.trade_date = x.two_day_before then 2
		when t.trade_date = x.three_day_before then 3
		when t.trade_date = x.four_day_before then 4
		when t.trade_date = x.five_day_before then 5   
		when t.trade_date = x.six_day_before then 6  
		when t.trade_date = x.seven_day_before then 7
		else 0
	end) as 'before_day'	
        
from	trade_raws_taipei as t
join	( select WDate, date_add(d.WDate, interval -1 day) as 'one_day_before', date_add(d.WDate, interval -2 day) as 'two_day_before', date_add(d.WDate, interval -3 day) as 'three_day_before', 
			date_add(d.WDate, interval -4 day) as 'four_day_before', date_add(d.WDate, interval -5 day) as 'five_day_before', 
            date_add(d.WDate, interval -6 day) as 'six_day_before', date_add(d.WDate, interval -7 day) as 'seven_day_before' 
			from	date_map as d
			where  (d.Holiday_Name = '清明' or d.Holiday_Name = '端午' or d.Holiday_Name = '中元' or d.Holiday_Name = '中秋' or d.Holiday_Name = '除夕') and (d.WDate between '2012-01-01' and '2020-05-15')) as x
on	t.trade_date = x.two_day_before or t.trade_date = x.three_day_before or 
t.trade_date = x.four_day_before or t.trade_date = x.five_day_before or t.trade_date = x.six_day_before or t.trade_date = x.seven_day_before
;


select 	*
from 	trade_raws_taipei t
join	date_map d
on		t.trade_date = d.WDate;	

select * from date_map where WDate between '2012-01-01' and '2020-05-15';

select * from trade_raws_taipei;

select 	distinct t.trade_date, t.avg_price, t.trade_amount,   
	(case 
		when t.trade_date = date_add(x.WDate, interval -1 day) then '-1'
		when t.trade_date = date_add(x.WDate, interval -2 day) then '-2'
		when t.trade_date = date_add(x.WDate, interval -3 day) then '-3'
		when t.trade_date = date_add(x.WDate, interval -4 day) then '-4'
		when t.trade_date = date_add(x.WDate, interval -5 day) then '-5'  
		when t.trade_date = date_add(x.WDate, interval -6 day) then '-6'
		when t.trade_date = date_add(x.WDate, interval -7 day) then '-7'
        
        when t.trade_date = date_add(x.WDate, interval 1 day) then '1'
        when t.trade_date = date_add(x.WDate, interval 2 day) then '2'
        when t.trade_date = date_add(x.WDate, interval 3 day) then '3'
        when t.trade_date = date_add(x.WDate, interval 4 day) then '4'
        when t.trade_date = date_add(x.WDate, interval 5 day) then '5'
        when t.trade_date = date_add(x.WDate, interval 6 day) then '6'
		when t.trade_date = date_add(x.WDate, interval 7 day) then '7'
	else '0'
	end) as 'about_holiday'	
        
from	trade_raws_taipei as t
join	( select WDate 
			from	date_map as d
			where  (d.Holiday_Name = '清明' or d.Holiday_Name = '端午' or d.Holiday_Name = '中元' or d.Holiday_Name = '中秋' or d.Holiday_Name = '除夕') and (d.WDate between '2012-01-01' and '2020-05-15')) as x
;