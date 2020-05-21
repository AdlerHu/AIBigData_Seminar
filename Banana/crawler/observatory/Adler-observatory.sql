CREATE TABLE Adler_observatory(
	`code`		varchar(10), 
	`name`		varchar(10),
    `city`      varchar(5),
    `start_date` date    
);

select * from Adler_observatory;

drop table Adler_observatory;

INSERT INTO adler_observatory (`code`, `name`, `city`, `start_date`) VALUES	('C1V570', '路專', '桃園市', '1992-06-27'); 

delete  from adler_observatory where `date` = '1992-06-27'; 

SET SQL_SAFE_UPDATES=0;

UPDATE mysql.user SET Grant_priv='Y', Super_priv='Y' WHERE User='root';
FLUSH PRIVILEGES;

alter user 'root'@'localhost' identified with mysql_native_password by 'root';

select count(*) from adler_observatory;