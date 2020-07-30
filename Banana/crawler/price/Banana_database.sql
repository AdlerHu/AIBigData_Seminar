CREATE TABLE banana(
	`b_id`					integer primary key auto_increment, 
	`transaction_date`		VARCHAR(20),
	`crop_number`			varchar(5),
    `crop_name`				varchar(20),
    `market_number`			varchar(5),
    `market_name`			varchar(20),
    `high_price`			varchar(10),
    `middle_price`			varchar(10),
    `low_price`				varchar(10),
    `avg_price`				varchar(10),
    `volume`				varchar(20)
);

select * from banana;

select * from banana where crop_name = '測試';

drop table banana;

INSERT INTO banana	(transaction_date, crop_number, crop_name, market_number, market_name, high_price, middle_price, low_price, avg_price, volume)
VALUES			('106.02.18', 'A1', '測試', '400', '台中市', '110.0', 82.4, 49.7, 85.3, 12740.0); 

delete  from banana where crop_name = '測試'; 

SET SQL_SAFE_UPDATES=0;

UPDATE mysql.user SET Grant_priv='Y', Super_priv='Y' WHERE User='root';
FLUSH PRIVILEGES;

alter user 'root'@'localhost' identified with mysql_native_password by 'root';

# Error Code: 1175. You are using safe update mode and you tried to update a table without a WHERE that uses a KEY -- column.  To disable safe mode, toggle the option in Preferences -> SQL Editor and reconnect.
