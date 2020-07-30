CREATE TABLE Adler_gdp(
	`annual`	int, 
    `season`	varchar(5),
    `start_date` date,
    `end_date`	 date,
    `gdp`		int,
    `egr`		float,
    
    PRIMARY KEY (Annual, Season)
);

select * from Adler_gdp;

drop table Adler_gdp;

INSERT INTO Adler_gdp	(Annual, Season, GDP, YOY) VALUES ('200年',  '第1季','1992-06-27', '2020-15-05',  '3573234', '1.09'); 

delete  from Adler_gdp where Annual = '200年'; 

SET SQL_SAFE_UPDATES=0;

UPDATE mysql.user SET Grant_priv='Y', Super_priv='Y' WHERE User='root';
FLUSH PRIVILEGES;

alter user 'root'@'localhost' identified with mysql_native_password by 'root';

# Error Code: 1175. You are using safe update mode and you tried to update a table without a WHERE that uses a KEY -- column.  To disable safe mode, toggle the option in Preferences -> SQL Editor and reconnect.

select count(*) from banana