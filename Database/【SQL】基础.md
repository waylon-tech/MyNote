## 目录
[toc]

## 1 SQL简介
SQL 集数据定义语言（DDL），数据操纵语言（DML），数据控制语言（DCL）功能于一体。

### 1.1 查询与更新指令
构成了SQL的DML部分：

* `SELECT` - 从数据库表中获取数据
* `UPDATE` - 更新数据库表中的数据
* `DELETE` - 从数据库表中删除数据
* `INSERT INTO` - 向数据库表中插入数据

### 1.2 表格与数据库操作
**（1）语法**

以及表关系的指令构成了SQL的DDL部分，SQL中最重要的DDL语句：

* `SHOW DATABASES` - 显示当前所有数据库
* `SHOW TABLES` - 显示当前所有的表格
* `USE 数据库名称` - 切换/选择数据库
* `CREATE DATABASE` - 创建数据库
* `ALTER DATABASE` - 修改数据库
* `CREATE TABLE` - 创建数据库表
* `ALTER TABLE` - 变更数据库表
* `DROP TABLE` - 删除数据库表
* `CRAETE INDEX` - 创建索引
* `DROP INDEX` - 删除索引

详细语法在后面。

**（2）完整性约束条件**

* `NOT NULL` : 非空约束，指定某列不能为空
* `UNIQUE` : 唯一约束，指定某列或者几列组合不能重复
* `PRIMARY KEY` : 主键，指定该列的值可以唯一地标识该列记录
* `FOREIGN KEY` : 外键，指定该行记录从属于主表中的一条记录，主要用于参照完整性
* `DEFAULT` : 默认值，创建列时可以指定默认值，当插入数据时如果未主动设置，则自动添加默认值
* `AUTO_INCREMENT` : 约束字段为自动增长，被约束的字段必须同时被`key`约束

详细语法在后面。

### 1.3 一些要知道的东西

* 选择所有的内容，可以使用通配符`*`
* 在SQL中，使用单引号`'`来环绕文本值

**(1) 登陆mySql：**

```shell
> mysql -u 用户名 -p
你的密码
```

**(2) 启动mysql服务：**

在cmd命令行下，输入：`net start MySQL`

注意：这里的mysql服务名随着计算机而异

**(3) MySQL的密码修改：**

windows打开my.ini文件，linux打开my.cnf文件，在mysqld下面添加：

```ini
skip-grant-tables
```

然后保存退出。<u>如果该方法导致服务无法启动，则用第二种方法。第二种跳过密码检验的方法在下一个点。</u>

重启MySQL服务，我的电脑右击 -> 管理 -> 应用服务 -> 服务，然后运行`cmd`。

输入：

```shell
mysql
# 或者
mysql -u root -p
```

直接回车就可以进入。

进入mysql数据库：

```shell
mysql> use mysql;
```

给root用户设置新密码：

```mysql
# 如果口令列名为 password
update mysql.user set password=password("自己的密码") where user="root" and Host = 'localhost';
# 如果口令列名为 authentication_string
update mysql.user set authentication_string="" where User="root" and Host = "localhost"; # 先置空，再退出skip-grant-tables，重启服务后来修改密码
# MySQL8.0 使用新的函数
ALTER USER 'root'@'localhost' IDENTIFIED WITH MYSQL_NATIVE_PASSWORD BY '新密码';
```

刷新数据库：

```shell
mysql> flush privileges;
```

退出mysql：

```shell
mysql> quit
```

修改一下my.ini文件，将刚加入的`skip-grant-tables`删除，保存退出再重启mysql服务。

**（4）第二种跳过密码检验的方法**

* 管理员权限登陆cmd
* 停止mysql相关的服务
  * 我的电脑右击 -> 管理 -> 应用服务 -> 服务
  * win+R --> services.msc --> 回车
* `mysqld --console --skip-grant-tables --shared-memory`停止密码检查
* 另开cmd，使用mysql登录（这里暂时不用重启服务，已经有一个临时的了），然后回到上面的设置新密码做法，并刷新权限
* 重启服务登录

## 2 SQL基础语法（DML)

### 2.1 `SELECT`
该**语句**用于从表中选取数据。

语法：

```mysql
SELECT 列名称 FROM 表名称
```

### 2.2 `DISTINCT`

该语句用于返回唯一不同的值。

语法：

```mysql
SELECT DISTINCT 列名称 FROM 表名称
```

### 2.3 `WHERE`
该**子句**允许有条件地从表中选取数据。

语法：

```mysql
SELECT 列名称 FROM 表名称 WHERE 列 运算符 值
```

关于运算符：

* `=` - 等于
* `<>` - 不等于
* `>` - 大于
* `<` - 小于
* `>=` - 大于等于
* `<=` - 小于等于
* `BETWEEN...AND` - 在某范围内（下见）
* `NOT BETWEEN...AND` - 在某范围内（下见）
* `IN <值表>` - 用于规定多个值（下见）
* `NOT IN <值表>`  - 用于规定多个值（下见）
* `LIKE` - 搜索某种模式（下见）

### 2.4 `AND`和`OR`
该**运算符**可在`WHERE`子句中把两个或多个条件结合。

使用圆括号复合组合。

实例：

```mysql
SELECT * FROM Persons WHERE (FirstName='Thomas' OR FirstName='William') AND LastName='Carter'
```

### 2.5 `ORDER BY`
该**语句**用于根据指定的列对结果排序，默认升序。

语法：

```mysql
结果集 ORDER BY 列 DESC
// 若使用降序，可在末尾添加`DESC`关键字。
```

实例：

```mysql
SELECT Company, OrderNumber FROM Orders ORDER BY Company DESC, OrderNumber
```

### 2.6 `INSERT INTO`
该**语句**用于向表格中插入新的行。

语法：

```mysql
INSERT INTO 表名称 VALUES (值1, 值2, ...)

INSERT INTO 表名称 (列1, 列2, ...) VALUES (值1, 值2, ...) // 指定列，对应值
```

实例：

```mysql
INSERT INTO Persons (LastName, Address) VALUES ('Wilson', 'Champs-Elysees')
```

### 2.7 `UPDATE`
该**语句**用于修改表中的数据。

语法：

```mysql
UPDATE 表名称 SET 列名称 = 新值 WHERE 列名称 = 某值
```

实例：

```mysql
UPDATE Person SET Address = 'Zhongshan 23', City = 'Nanjing' WHERE LastName = 'Wilson'
```

### 2.8 `DELETE`
该**语句**用于删除表中的行。

语法：

```mysql
DELETE FROM 表名称 WHERE 列名称 = 值
```

实例：

```mysql
DELETE FROM Person WHERE LastName = 'Wilson'
```

## 3 SQL高级语法（DML+DDL）

### 3.1 `TOP`
该**子句**用于规定要返回的记录的数目。

语法：

```mysql
SELECT TOP 数目/百分比 列名称 FROM 表名称
```

注：在Oracle中的语法为：

```mysql
SELECT 列名称 FROM 表名称 WHERE ROWNUM <= 数目
```

实例：

```mysql
SELECT TOP 50 PERCENT * FROM Persons
```

### 3.2 `LIKE`
该**操作符**用于在`WHERE`子句中搜索列的指定模式。

语法：

```mysql
SELECT 列名称 FROM 表名称
WHERE 列名称 LIKE 模式
// 使用`NOT`关键字，选取补集
```

关于模式：在SQL中，通配符与`LIKE`运算符一起使用，有如下：

* `%` - 替换一个或多个字符
* `_` - 仅替换一个字符
* `[charlist]` - 字符列表中的任何单一字符
* `[^charlist]` - 不在字符列表的任何单一字符

实例：

```mysql
SELECT * FROM Persons WHERE City LIKE '[!ALN]%'
```

### 3.3 `IN`
该**操作符**允许在`WHERE`子句中规定多个值。

语法：

```mysql
SELECT 列名称 FROM 表名称
WHERE 列名称 IN (值1, 值2, ...)
```

实例：

```mysql
	SELECT * FROM Persons WHERE LastName IN ('Adams','Carter')
```

### 3.4 `BETWEEN...AND`
该**操作符**会选取介于两数值/文本/日期之间的数据范围。

语法：

```mysql
SELECT 列名称 FROM 表名称
WHERE 列名称
BETWEEN 值1 AND 值2
// 使用`NOT`关键字，选取补集
```

实例：

```mysql
SELECT * FROM Persons WHERE LastName NOT BETWEEN 'Adams' AND 'Carter'
```

### 3.5 `AS`
该**语句**用于为列名称、表名称指定别名（Alias）。

语法：

```mysql
// 表名称别名：
SELECT 列名称 FROM 表名称 AS 别名

// 列名称别名：
SELECT 列名称 AS 别名 FROM 表名称
```

实例：

```mysql
// 表名称别名
SELECT po.OrderID, p.LastName, p.FirstName FROM Persons AS p, Product_Orders AS po WHERE p.LastName='Adams' AND p.FirstName='John'

// 列名称别名
SELECT LastName AS Family, FirstName AS Name FROM Persons
```

### 3.6 `JOIN`
该**关键字**用于从两个或多个表中获取结果。

* `JOIN` - 如果表中至少有一个匹配，则返回行
* `INNER JOIN` - 与`JOIN`相同
* `LEFT JOIN` - 即使右表中没有匹配，也从左表返回所有行
* `RIGHT JOIN` - 即使左表中没有匹配，也从右表返回所有行
* `FULL JOIN` - 只要有一个表存在匹配，就返回行

语法：

```mysql
SELECT 列名称 FROM 表名称1
JOIN 表名称2
ON 表名称1.列名称=表名称2.列名称
```

实例：

```mysql
SELECT Persons.LastName, Persons.FirstName, Orders.OrderNo FROM Persons INNER JOIN Orders ON Persons.Id_P=Orders.Id_P ORDER BY Persons.LastName
```

### 3.7 `UNION`
该**操作符**用于合并两个或多个`SELECT`语句结果集。

语法：

```mysql
SELECT 列名称 FROM 表名称
UNION
SELECT 列名称 FROM 表名称
// 如果允许重复，使用`UNION ALL`
```

实例：

```mysql
SELECT E_Name FROM Employees_China UNION ALL SELECT E_Name FROM Employees_USA
```

### 3.8 `SELECT INTO`
该**语句**从一个表中选取数据，然后插入另一个表中。

语法：

```mysql
SELECT 列名称
INTO 新表名称 IN 外数据库
FROM 旧表名称
```

实例：

```mysql
SELECT Persons.LastName,Orders.OrderNo
INTO Persons_Order_Backup
FROM Persons
INNER JOIN Orders
ON Persons.Id_P=Orders.Id_P
```

### 3.9 `CREATE DATABASE`
该**语句**用于创建数据库。

语法：

```mysql
CREATE DATABASE 数据库名称
```

实例：

```mysql
CREATE DATABASE my_db
```

### 3.10 `CREATE TABLE`
该**语句**用于创建数据库中的表。

另外，每一个基本表都属于某一个模式，一个模式包含多个基本表。这里暂时不做探讨。

语法：

```mysql
CREATE TABLE <表名>(
    <列名> <数据类型>[ <列级完整性约束条件> ]
    [,<列名> <数据类型>[ <列级完整性约束条件> ] ]
    ...
    [,<表级完整性约束条件> ]
);
```

关于数据类型的简单补充：

* 整数：
	* `integer(size)` - 容纳整数，括号内规定数字的最大位数
	* `int(size)` - 容纳整数，括号内规定数字的最大位数
	* `smallint(size)` - 容纳整数，括号内规定数字的最大位数
	* `tinyint(size)` - 容纳整数，括号内规定数字的最大位数
* 小数：
  * `double(size, d)` - 带有浮动小数点的大数字。size规定最大位数。d规定小数点右侧的最大位数
  * `decimal(size, d)` - 作为字符串存储的 DOUBLE 类型，允许固定的小数点
* 字符串：
  * `char(size)` - 容纳固定长度的字符串，括号内规定字符串的长度
  * `varchar(size)` - 容纳可变长度的字符串，括号内规定字符串的最大长度
* 日期：
  * `data(yyyymmdd)` - 容纳日期

实例：

```mysql
USE puresakura;

CREATE TABLE `readers` (
    `rid` VARCHAR(50) NOT NULL,
    `rname` VARCHAR(50) NOT NULL,
    `rsex` TINYINT NOT NULL,
    `remail` VARCHAR(50) NOT NULL,
    `rrole` VARCHAR(50) NOT NULL,
    `radmin` TINYINT NULL DEFAULT 0,
    PRIMARY KEY (`rid`)
)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

CREATE TABLE `books` (
    `bid` VARCHAR(50) NOT NULL,
    `btitle` VARCHAR(50) NOT NULL,
    `bauthor` VARCHAR(50) NOT NULL,
    `bpublisher` VARCHAR(50) NOT NULL,
    `bpublished_at` REAL NULL DEFAULT 0,
    `bsort` VARCHAR(50) NOT NULL,
    `bread_times` INT NULL DEFAULT 0,
    `bexits` TINYINT NOT NULL,
    PRIMARY KEY (`bid`)
)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

CREATE TABLE `borrows` (
    `id` VARCHAR(50) NOT NULL,
    `rid` VARCHAR(50) NOT NULL,
    `bid` VARCHAR(50) NOT NULL,
    `bborrow_time` REAL NULL DEFAULT '0',
    `bdue_time` REAL NULL DEFAULT '0',
    `breturn_time` REAL NULL DEFAULT '0',
    `bcomment` VARCHAR(100) NOT NULL,
    PRIMARY KEY (`id`),
    CONSTRAINT `rid`
        FOREIGN KEY (`rid`)
        REFERENCES `readers` (`rid`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `bid`
        FOREIGN KEY (`bid`)
        REFERENCES `books` (`bid`)
        ON DELETE CASCADE
        ON UPDATE CASCADE
)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;
```

### 3.11 `CONSTRAINTS`
约束**关键字**用于限制加入表的数据类型，有以下几种：

* **`NOT NULL`** - 非空约束，指定某列不能为空

语法：

```mysql
列名称 数据类型 NOT NULL
```

实例：

```mysql
create table t12 (
    id int not null
);
```

* **`UNIQUE`** - 唯一约束，指定某列或者几列组合不能重复

语法：

```mysql
// 创建
列名称 数据类型
UNIQUE (列名称)				// MySQL
列名称 数据类型 UNIQUE		// SQL Server / Oracle / MS Access

// 添加
ALTER TABLE 表名称
ADD UNIQUE (列名称)			// MySQL / SQL Server / Oracle / MS Access

// 命名
CONSTRAINT 约束名称 UNIQUE (列名称1, 列名称2, ...)		// 创建，MySQL / SQL Server / Oracle / MS Access
ALTER TABLE 表名称
ADD CONSTRAINT 约束名称 UINQUE (列名称1, 列名称2, ...)	// 添加，MySQL / SQL Server / Oracle / MS Access

// 撤销
ALTER TABLE 表名称
DROP INDEX 约束名称			// MySQL
ALTER TABLE 表名称
DROP CONSTRAINT 约束名称		// SQL Server / Oracle / MS Access
```

实例：

```mysql
# 方法一
create table department1(
    id int,
    name varchar(20) unique, # 列级完整性约束
    comment varchar(100)
);

# 方法二
create table department2(
    id int,
    name varchar(20),
    comment varchar(100),
    unique(name) # 表级完整性约束
);

# not null 和 unique 的结合
create table t1(
    id int not null unique # 结合
);

# 联合唯一
create table service(
    id int primary key auto_increment,
    name varchar(20),
    host varchar(15) not null,
    port int not null,
    unique(host,port) # 联合唯一
);
```

* **`KEY`** - 普通索引，唯一的作用是加快对数据的访问速度。

语法：

```mysql
KEY 'idx_t_qx_mx_qxlx'('qxlx')
```

表示在变量`qxlx`上有索引，所有名字为`idx_t_qx_mx_qxlx`.

可以执行`show index from table1`查看表上的索引。

* **`PRIMARY KEY`** - 唯一标识数据库表中的每条记录,区别：(1)主键值不为NULL (2)主键唯一

语法：

```mysql
// 创建
列名称 数据类型 NOT NULL
PRIMARY KEY (列名称)					// MySQL
列名称 数据类型 NOT NULL PRIMARY		// SQL Server / Oracle / MS Access

// 添加
ALTER TABLE 表名称
ADD PRIMARY KEY (列名称)				// MySQL / SQL Server / Oracle / MS Access

// 命名
CONSTRAINT 约束名称 PRIMARY KEY (列名称1, 列名称2, ...)		// 创建，MySQL / SQL Server / Oracle / MS Access
ALTER TABLE 表名称
ADD CONSTRAINT 约束名称 PRIMARY KEY (列名称1, 列名称2, ...)	// 添加，MySQL / SQL Server / Oracle / MS Access

// 撤销
ALTER TABLE 表名称
DROP PRIMARY KEY			// MySQL
ALTER TABLE 表名称
DROP CONSTRAINT 约束名称		// SQL Server / Oracle / MS Access
```

实例：

```mysql
# 方法一：not null + unique
create table department1(
    id int not null unique, # 主键
    name varchar(20) not null unique,
    comment varchar(100)
);

# 方法二：在某一个字段后用 primary key
create table department2(
    id int primary key, # 主键
    name varchar(20),
    comment varchar(100)
);

# 方法三：在所有字段后单独定义 primary key
create table department3(
    id int,
    name varchar(20),
    comment varchar(100),
    primary key(id,name) # 主键
);
    
# 方法四：给已经建成的表添加主键约束
alter table department4 modify id int primary key;
```

* **`FOREIGN KEY`** - 一个表中的FROEING KEY指向另一个表中的PRIMARY KEY，用于预防破坏表间连接，防止非法数据插入外键列

语法：

```mysql
// 创建
列名称 数据类型
FOREIGN KEY (列名称) REFERENCES 外表名称(外列名称)			// MySQL
列名称 数据类型 FOREIGN KEY REFERENCES 外表名称(外列名称)	// SQL Server / Oracle / MS Access

// 添加
ALTER TABLE 表名称
ADD FOREIGN KEY (列名称) REFERENCES 外表名称(外列名称)		// MySQL / SQL Server / Oracle / MS Access

// 命名
CONSTRAINT 约束名称 FOREIGN KEY (列名称) REFERENCES 外表名称(外列名称)		// 创建，MySQL / SQL Server / Oracle / MS Access
ALTER TABLE 表名称
ADD CONSTRAINT 约束名称 FOREIGN KEY (列名称) REFERENCES 外表名称(外列名称）	// 添加，MySQL / SQL Server / Oracle / MS Access

// 撤销
ALTER TABLE 表名称
DROP FOREIGN KEY 约束名称	// MySQL
ALTER TABLE 表名称
DROP CONSTRAINT 约束名称		// SQL Server / Oracle / MS Access
```

实例：

```mysql
create table departments (
    dep_id int(4),
    dep_name varchar(11)
);

create table staff_info (
    s_id int,name varchar(20),
    dep_id int not null unique, # 当设置字段为unique唯一字段时，设置该字段为外键成功
    foreign key(dep_id)
    references departments(dep_id)
);
```

* **`CHECK`** - 用于限制列中的值的范围

语法：

```mysql
// 创建
列名称 数据类型 
CHECK (列名称 运算符 值)					// MySQL
列名称 数据类型 CHECK (列名称 运算符 值)	// SQL Server / Oracle / MS Access

// 添加
ALTER TABLE 表名称
ADD CHECK (列名称 运算符 值)		// MySQL / SQL Server / Oracle / MS Access

// 命名
CONSTRAINT 约束名称 CHECK (列名称 运算符 值)		// 创建，MySQL / SQL Server / Oracle / MS Access
ALTER TABLE 表名称
ADD CONSTRAINT 约束名称 CHECK (列名称 运算符 值) 	// 添加，MySQL / SQL Server / Oracle / MS Access

// 撤销
ALTER TABLE 表名称
DROP CHECK 约束名称			// MySQL
ALTER TABLE 表名称
DROP CONSTRAINT 约束名称		// SQL Server / Oracle / MS Access
```

* **`DEFAULT`** - 用于向列中插入默认值

语法：

```mysql
// 创建
列名称 数据类型 DEFAULT 默认值	// MySQL / SQL Server / Oracle / MS Access

// 添加
ALTER TABLE 表名称
ALTER 列名称 SET DEFAULT 默认值			// MySQL
ALTER TABLE 表名称
ALTER COLUMN 列名称 SET DEFAULT 默认值	// SQL Server / Oracle / MS Access

// 撤销
ALTER TABLE 表名称
ALTER 列名称 DROP DEFAULT			// MySQL
ALTER TABLE 表名称
ALTER COLUMN 列名称 DROP DEFAULT		// SQL Server / Oracle / MS Access
```

实例：

```mysql
create table t13 (
    id1 int not null,
    id2 int not null default 222
);
```

* **`AUTO_INCREMENT`** - 约束字段为自动增长，被约束的字段必须同时被`key`约束
  * `AUTO_INCREMENT`是数据列的一种属性，只适用于整数类型数据列
  * `AUTO_INCREMENT`数据列必须有唯一索引
  * `AUTO_INCREMENT`数据列必须具备`NOT NULL`属性

实例：

```mysql
create table student(
    id int primary key auto_increment, # 不指定id，则自动增长
    name varchar(20),
    sex enum('male','female') default 'male'
)
auto_increment=3; # 起始值
```

### 3.12 `CREATE INDEX`

该**语句**用于在表中创建索引。

语法：

```mysql
// 创建
CREATE INDEX 索引名称
ON 表名称 (列名称)

// 创建唯一索引
CREATE UNIQUE INDEX 索引名称
ON 表名称 (列名称)
```

实例：

```mysql
CREATE INDEX PersonIndex
ON Person (LastName DESC, FirstName)
```

### 3.13 `DROP`
该**语句**用于删除索引、表和数据库。 

* 删除索引

  ```mysql
  DROP INDEX 索引名称 ON 表名称		// Microsoft SQLJet、Microsoft Access
  
  DROP INDEX 表名称.索引名称		// MS SQL Server
  
  DROP INDEX 索引名称				// IBM DB2、Oracle
  
  ALTER TABLE 表名称 DROP INDEX 索引名称	// MySQL
  ```

* 删除表

  ```mysql
  DROP TABLE 表名称
  ```

* 删除数据库

  ```mysql
  DROP DATABASE 数据库名称
  ```

* 清空表内数据

	```mysql
	TRUNCATE TABLE 表名称
	```

### 3.14 `ALTER`
该**语句**用于在已有的表中添加、修改或删除列。

* 添加列

  ```mysql
  ALTER TABLE 表名称
  ADD 列名称 数据类型
  ```

* 删除列

  ```mysql
  ALTER TABLE 表名称
  DROP COLUMN 列名称
  ```

* 改变数据类型

	```mysql
	ALTER TABLE 表名称
	ALTER COLUMN 列名称 数据类型
	```

### 3.15 `AUTO_INCREMENT`
该**字段**会在新记录插入表中时生成一个唯一的数字。

* 用于MySQL的语法：

  ```mysql
  CREATE TABLE 表名称
  (
  列名称 数据类型 AUTO-INCREMENT,
  ...
  )
  
  // 若要改变起始值
  ALTER TABLE 表名称 AUTO_INCREMENT=起始值
  ```

* 用于SQL Server的语法：

  ```mysql
  CREATE TABLE 表名称
  (
  列名称 数据类型 IDENTITY
  ...
  )
  
  // 若要改变起始值，将IDENTITY改为
  IDENTITY(起始值, 间隔值)
  ```

* 用于ACCESS的语法：

  ```mysql
  CREATE TABLE 表名称
  (
  列名称 数据类型 AUTOINCREMENT,
  ...
  )
  
  // 若要改变起始值，将AUTOINCREMENT改为
  AUTOINCREMENT(起始值, 间隔值)
  ```

* 用于ORACLE的语法：

	```mysql
	CREATE SEQUENCE 序列名称
	MINVALUE 最小值
	START WITH 起始值
	INCREMENT BY 间隔值
	CACHE 缓存数目
	
	INSERT INTO 表名称 (列名称1, 列名称2, ...)
	VALUES (序列名称.nextval, 值2, ...)
	```

### 3.16 `VIEW`
视图是基于SQL语句的结果集的可视化表。

用于向用户呈现定制数据。

语法：

```mysql
// 创建视图
CREATE VIEW 视图名称 AS
SELECT 列名称
FROM 表名称
WHERE 条件

// 更新视图
SQL CREATE OR REPLACE VIEW Syntax
CREATE OR REPLACE VIEW 视图名称 AS
SELECT 列名称
FROM 表名称
WHERE 条件

// 撤销视图
SQL DROP VIEW Syntax
DROP VIEW 视图名称
```

实例：

```mysql
CREATE VIEW [Current Product List] AS
SELECT ProductID,ProductName
FROM Products
WHERE Discontinued=No

SELECT * FROM [Current Product List]

CREATE VIEW [Current Product List] AS
SELECT ProductID,ProductName,Category
FROM Products
WHERE Discontinued=No
```

### 3.17 `DATE`函数
* 函数
	* MySQL
		* `NOW()` - 返回当前的日期和时间 
		* `CURDATE()` - 返回当前的日期 
		* `CURTIME()` - 返回当前的时间 
		* `DATE()` - 提取日期或日期/时间表达式的日期部分 
		* `EXTRACT()` - 返回日期/时间按的单独部分 
		* `DATE_ADD()` - 给日期添加指定的时间间隔 
		* `DATE_SUB()` - 从日期减去指定的时间间隔 
		* `DATEDIFF()` - 返回两个日期之间的天数 
		* `DATE_FORMAT()` - 用不同的格式显示日期/时间
	* SQL Server
		* `GETDATE()` 返回当前日期和时间 
		* `DATEPART()` - 返回日期/时间的单独部分 
		* `DATEADD()` - 在日期中添加或减去指定的时间间隔 
		* `DATEDIFF()` - 返回两个日期之间的时间 
		* `CONVERT()` - 用不同的格式显示日期/时间 

* 数据类型
	* MySQL
		* DATE - 格式 `YYYY-MM-DD` 
		* DATETIME - 格式: `YYYY-MM-DD HH:MM:SS`
		* TIMESTAMP - 格式: `YYYY-MM-DD HH:MM:SS`
		* YEAR - 格式 `YYYY` 或 `YY`
	* SQL Server
		* DATE - 格式 `YYYY-MM-DD` 
		* DATETIME - 格式: `YYYY-MM-DD HH:MM:SS` 
		* SMALLDATETIME - 格式: `YYYY-MM-DD HH:MM:SS` 
		* TIMESTAMP - 格式: 唯一的数字 
	* 注意
		* 不涉及时间部分可以轻松地比较两个日期
		* 为了查询简单且更易维护，建议不要在日期中使用时间部分

### 3.18 `NULL`
该**操作符**表示遗漏的位置数据。用于未知的或不使用的值的占位符。

* 操作符
	* `IS NULL` - 确认NULL值
	* `IS NOT NULL` - 确认非NULL值

* 函数处理
	* SQL Server / MS Access
		* ISNULL(列名称, 替换值)
	* Oracle
		* NVL(列名称, 替换值)
	* MySQL
		* IFNULL(列名称, 替换值)
		* COALESCE(列名称, 替换值)

### 3.19 附录
* SQL 数据类型
	* 参见 Library-WEB-W3cSchool
* SQL 服务器
	* DBMS - 数据库管理系统（Database Management System）
		* 数据库管理系统是一种可以访问数据库中数据的计算机程序。
		* DBMS 使我们有能力在数据库中提取、修改或者存贮信息。
		* 不同的 DBMS 提供不同的函数供查询、提交以及修改数据。
	* RDBMS - 关系数据库管理系统（Relational Database Management System）
		* 关系数据库管理系统 (RDBMS) 也是一种数据库管理系统，其数据库是根据数据间的关系来组织和访问数据的。
		* 20 世纪 70 年代初，IBM 公司发明了 RDBMS。
		* RDBMS 是 SQL 的基础，也是所有现代数据库系统诸如 Oracle、SQL Server、IBM DB2、Sybase、MySQL 以及 Microsoft Access 的基础。

