CREATE TABLE Adler_weather(
	`station`				varchar(10), 
	`date`            		date,
    `hour`					varchar(4),
    `pressure`				VARCHAR(10),
	`temperature`			varchar(5),
    `humidity`				varchar(5),
    `wind_speed`			varchar(5),
    `wind_direction`		varchar(10),
    `precipitation`			varchar(30),
    
    primary key (station, `date`, `hour`)
);

select * from Adler_weather;

drop table Adler_weather;

INSERT INTO Adler_weather	(`station`, `date`, `hour`, `pressure`, `temperature`, `humidity`, `wind_speed`, `wind_direction`, `precipitation`)
VALUES			('C0V440', '1992-06-06','66', '1017.1', '14.2', '81', '0.5', '162', '0.0'); 

delete  from Adler_weather where `hour` = '66'; 

SET SQL_SAFE_UPDATES=0;

UPDATE mysql.user SET Grant_priv='Y', Super_priv='Y' WHERE User='root';
FLUSH PRIVILEGES;

alter user 'root'@'localhost' identified with mysql_native_password by 'root';

# Error Code: 1175. You are using safe update mode and you tried to update a table without a WHERE that uses a KEY -- column.  To disable safe mode, toggle the option in Preferences -> SQL Editor and reconnect.
