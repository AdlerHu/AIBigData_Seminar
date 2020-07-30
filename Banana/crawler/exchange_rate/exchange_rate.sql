CREATE TABLE Adler_rate(
	`date`		date primary key, 
	`rate`		float
);

select * from Adler_rate;

drop table Adler_rate;

INSERT INTO Adler_rate	(`date`, `rate`) VALUES	('1992-06-27', 99.98); 

delete  from Adler_rate where `date` = '1992-06-27'; 

SET SQL_SAFE_UPDATES=0;

UPDATE mysql.user SET Grant_priv='Y', Super_priv='Y' WHERE User='root';
FLUSH PRIVILEGES;

alter user 'root'@'localhost' identified with mysql_native_password by 'root';

# Error Code: 1175. You are using safe update mode and you tried to update a table without a WHERE that uses a KEY -- column.  To disable safe mode, toggle the option in Preferences -> SQL Editor and reconnect.
