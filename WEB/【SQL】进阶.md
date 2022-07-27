## 目录

[toc]

## 1 数据定义

### 1.1 建库

* `CREATE DATABASE`：该**语句**用于创建数据库

语法：

```mysql
CREATE DATABASE 数据库名称
```

实例：

```mysql
CREATE DATABASE my_db
```

### 1.2 建表

* `CREATE TABLE`：该**语句**用于创建数据库中的表

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

### 1.3 约束

约束**关键字**用于限制加入表的数据类型，有以下几种：

#### 1.3.1 `NOT NULL`

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

#### 1.3.2 `UNIQUE`

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

#### 1.3.3 `KEY`

* **`KEY`** - 普通索引，唯一的作用是加快对数据的访问速度。

语法：

```mysql
KEY 'idx_t_qx_mx_qxlx'('qxlx')
```

表示在变量`qxlx`上有索引，索引名字为`idx_t_qx_mx_qxlx`.

可以执行`show index from table1`查看表上的索引。

#### 1.3.4 `PRIMARY KEY`

* **`PRIMARY KEY`** - 唯一标识数据库表中的每条记录，区别：(1)主键值不为NULL (2)主键唯一

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

#### 1.3.5 `FOREING KEY`

* **`FOREIGN KEY`** - 一个表中的`FOREING KEY`指向另一个表中的`PRIMARY KEY`，用于预防破坏表间连接，防止非法数据插入外键列

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
    foreign key (dep_id)
    references departments(dep_id)
);
```

#### 1.3.6 `CHECK`

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

#### 1.3.7 `DEFAULT`

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

#### 1.3.8 `AUTO_INCREMENT`

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

## 2 数据查询

```mysql
SELECT [ALL|DISTINCT] <目标列表达式>[,<目标列表达式>] ...
FROM <表名或视图名> [别名]
	 [,<表名或视图名> [别名]]...
	 |(SELECT 语句) [AS]<别名>
[ WHERE <条件表达式> ]
[ GROUP BY <列名1> [ HAVING <条件表达式> ] ]
[ ORDER BY <列名2> [ ASC|DESC ] ];
```

* `SELECT`子句：指定要显示的属性列
* `FROM`子句：指定查询对象（基本表或视图）
* `WHERE`子句：指定查询条件
* `GROUP BY`子句：对查询结果按指定列的值分组，该属性列值相等的元组为一个组（通常会在每组中作用聚集函数）
* `HAVING`短语：只有满足指定条件的组才予以输出
* `ORDER BY`子句：对查询结果表按指定列值的升序或降序排序

### 2.1 单表查询

#### 2.1.1 选择表中的若干列

**（1）查询指定列**

* 指定`<目标列表达式>`

```mysql
# 查询全体学生的学号与姓名
SELECT Sno, Sname
FROM Student;
```

**（2）查询全部列**

* 在`SELECT`关键字后面列出所有列名
* 将`<目标列表达式>`指定为`*`

```mysql
# 查询全体学生的详细记录（方法1）
SELECT Sno,Sname,Ssex,Sage,Sdept
FROM Student;

# 查询全体学生的详细记录（方法2）
SELECT *
FROM Student;
```

**（3）查询经过计算的值**

* 在`<目标列表达式>`中填充表达式

```mysql
# 查询全体学生的姓名、出生年份和所在的院系，要求用小写字母表示系名
SELECT Sname, 'Year of Birth: ', 2014-Sage, LOWER(Sdept) # a)字符串列 b)减法运算 c)函数运算
FROM Student;
# 输出：
Sname	'Year of Birth:'	2014-Sage	LOWER(Sdept)
李勇		Year of Birth:		1994		cs
刘晨 		Year of Birth:		1995		cs
王敏 		Year of Birth:		1996		ma
张立 		Year of Birth:		1995		cs

# 使用列别名改变查询结果的列标题
SELECT Sname NAME,'Year of Birth:' BIRTH, 2014-Sage BIRTHDAY,LOWER(Sdept) DEPARTMENT
FROM Student;
NAME		BIRTH			BIRTHDAY	DEPARTMENT
李勇		Year of Birth:		1994		cs
刘晨 		Year of Birth:		1995		cs
王敏 		Year of Birth:		1996		ma
张立 		Year of Birth:		1995		cs
```

#### 2.1.2 选择表中的若干元组

**（1）消除取值重复的行**

如果没有指定`DISTINCT`关键词，则缺省为`ALL`，即列出所有元组，不消除重复。

```mysql
# 指定 DISTINCT 关键词，去掉表中重复的行
SELECT DISTINCT Sno
FROM SC;
```

**（2）比较大小**

主要使用`WHERE`关键字的：

* `=` - 等于
* `<>` - 不等于
* `>` - 大于
* `<` - 小于
* `>=` - 大于等于
* `<=` - 小于等于

```mysql
# 查询计算机科学系全体学生的名单
SELECT Sname
FROM Student
WHERE Sdept=‘CS’;
```

**（3）确定范围**

主要使用`WHERE`关键字的：

* `BETWEEN...AND` - 在某范围内
* `NOT BETWEEN...AND` - 在某范围内

```mysql
# 查询年龄在20~23岁（包括20岁和23岁）之间的学生的姓名、系别和年龄
SELECT Sname, Sdept, Sage
FROM Student
WHERE Sage BETWEEN 20 AND 23;
```

**（4）确定集合**

主要使用`WHERE`关键字的：

* `IN <值表>` - 用于规定多个值
* `NOT IN <值表>`  - 用于规定多个值

```mysql
# 查询计算机科学系（CS）、数学系（MA）和信息系（IS）学生的姓名和性别
SELECT Sname, Ssex
FROM Student
WHERE Sdept IN ('CS','MA','IS');
```

**（5）字符匹配**

主要使用`WHERE`关键字的：

* `[NOT] LIKE '<匹配串>' [ESCAPE '<换码字符>']` - 搜索某种模式

  * `<匹配串>`可以是一个完整的字符串，也可以含有通配符`%`和` _`

  * `%`（百分号） 代表任意长度（长度可以为0）的字符串
  * `_`（下横线） 代表任意单个字符
  * 使用换码字符将通配符转义为普通字符

```mysql
# 查询名字中第2个字为"阳"字的学生的姓名和学号
SELECT * 
FROM Student
WHERE Sname LIKE '__阳%';

# 查询以"DB_"开头，且倒数第 3 个字符为 i 的课程的详细情况
SELECT *
FROM Course
WHERE Cname LIKE 'DB\_%i_ _' ESCAPE '\';
```

**（6）限定数量**

使用`LIMIT`关键字：

* `LIMIT` 子句可以被用于强制 `SELECT` 语句返回指定的记录数
* `LIMIT` 接受一个或两个数字参数。参数必须是整数常量
  * 如果只给定一个参数，它表示最多返回的行数
  * 如果给定两个参数，第一个参数指定第一个返回记录行的偏移量，第二个参数指定返回记录行的最大数目
  * 初始记录行的偏移量是 `0`（而不是 1）—— 为了与 `PostgreSQL` 兼容，`MySQL` 也支持句法： `LIMIT # OFFSET #`

```mysql
mysql> SELECT * FROM table LIMIT 5,10;  # 检索记录行 6-15，第一个参数偏移量指向的是目标位置的前一个位置，因此最小值是 0

# 为了检索从某一个偏移量到记录集的结束所有的记录行，可以指定第二个参数为 -1： 
mysql> SELECT * FROM table LIMIT 95,-1; # 检索记录行 96-last.

# 如果只给定一个参数，它表示返回最大的记录行数目： 
mysql> SELECT * FROM table LIMIT 5;     # 检索前 5 个记录行

# 换句话说，LIMIT n 等价于 LIMIT 0,n。


# 注1：LIMIT 后面不能接表达式，只能是一个变量或值
# 例如：选取第 N 高的薪水
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  set N = N-1; # 先运行就行
  RETURN (
      # Write your MySQL query statement below.
      select distinct Salary # 第 N 高的薪水
      from Employee
      order by Salary desc
      limit N,1
  );
END
```

**（7）多重条件查询**

使用逻辑运算符：`AND`和`OR`来连接多个查询条件。

```mysql
# 查询计算机系年龄在20岁以下的学生姓名
SELECT Sname
FROM Student
WHERE Sdept='CS' AND Sage<20;
```

#### 2.1.3 `ORDER BY`子句

该**语句**用于根据指定的列对结果排序，默认升序。

语法：

```mysql
结果集 ORDER BY 列 DESC
# 若使用降序，可在末尾添加`DESC`关键字
# 对于空值，排序时显示的次序由具体系统实现来决定
```

实例：

```mysql
SELECT *
FROM Student
ORDER BY Sdept, Sage DESC;
```

#### 2.1.4 聚集函数

* `COUNT(*)`：统计元组个数
* `COUNT([DISTINCT|ALL] <列名>)`：统计一列中值的个数
* `SUM([DISTINCT|ALL] <列名>)`：计算一列值的总和（此列必须为数值型）
* `AVG([DISTINCT|ALL] <列名>)`：计算一列值的平均值（此列必须为数值型）
* `MAX([DISTINCT|ALL] <列名>)`：求一列中的最大值
* `MIN([DISTINCT|ALL] <列名>)`：求一列中的最小值

```mysql
# 查询学生总人数
SELECT COUNT(*)
FROM Student;

# 查询学生201215012选修课程的总学分数
SELECT SUM(Ccredit)
FROM SC,Course
WHERE Sno='201215012' AND SC.Cno=Course.Cno;
```

#### 2.1.5 `GROUP BY`子句

该语句主要用于**细化聚集函数的作用对象**：

* 如果未对查询结果分组，聚集函数将作用于整个查询结果
* 对查询结果分组后，聚集函数将分别作用于每个组
* 按指定的一列或多列值分组，值相等的为一组

**【注1】**`WHERE`子句中是不能用聚集函数作为条件表达式，要使用聚集函数作为条件，必须用`GROUP BY`+`HAVING`。

**【注2】**`HAVING`短语与`WHERE`子句的区别：

* 作用对象不同
* `WHERE`子句作用于基表或视图，从中选择满足条件的元组
* `HAVING`短语作用于组，从中选择满足条件的组

```mysql
# 求各个课程号及相应的选课人数
SELECT Cno，COUNT(Sno) # 分组，然后计算人数
FROM SC
GROUP BY Cno;

# 查询平均成绩大于等于90分的学生学号和平均成绩
SELECT Sno, AVG(Grade)
FROM SC
GROUP BY Sno
HAVING AVG(Grade)>=90; # 必须用 HAVING
```

#### 2.1.6 窗口函数

* `RANK` 函数：为结果集的分区中的每一行分配一个排名。行的等级由一加上前面的等级数指定，同值后不连续。

  ```mysql
  RANK() OVER (
      PARTITION BY <expression>[{,<expression>...}]
      ORDER BY <expression> [ASC|DESC], [{,<expression>...}]
  )
  ```

  * 首先，`PARTITION BY` 子句将结果集划分为分区，**可以不指定**，此时全表为一个分区。`RANK()` 功能在分区内执行，并在<u>跨越分区边界时重新初始化</u>。
  * 其次，`ORDER BY` 子句按一个或多个列或表达式对分区内的行进行排序。
  * 注意，与 `ROW_NUMBER()` 函数不同，`RANK()` 函数**并不总是返回连续的整数**。行的值相同时，<u>跳过当前等级，取上一等级</u>。

  ```mysql
  # 按年分区，根据销售额进行排名
  SELECT								+----------------+-------------+--------+------------+
  	sales_employee,					| sales_employee | fiscal_year | sale   | sales_rank |
  	fiscal_year,					+----------------+-------------+--------+------------+
  	sale,							| John           |        2016 | 200.00 |          1 |
  	RANK( ) OVER ( 					| Alice          |        2016 | 150.00 |          2 |
  		PARTITION BY fiscal_year 	| Bob            |        2016 | 100.00 |          3 |
  		ORDER BY sale DESC			| Bob            |        2017 | 150.00 |          1 | # 跨越分区边界时重新初始化
  	) sales_rank 					| John           |        2017 | 150.00 |          1 | # 跳过当前等级，取上一等级
  FROM								| Alice          |        2017 | 100.00 |          3 |
  	sales; 							+----------------+-------------+--------+------------+
  ```

* `DENSE_RANK` 函数：为结果集的分区中的每一行分配一个排名。行的等级由一加上前面的等级数指定，同值后保持连续。

### 2.2 连接查询

#### 2.2.1 等值与非等值连接查询

使用连接运算符`=`，只有满足列相等的元组才会被采纳。

```mysql
# 查询每个学生及其选修课程的情况
SELECT Student.*, SC.*
FROM Student, SC # 笛卡尔积
WHERE Student.Sno = SC.Sno; # 过滤设定条件
```

连接操作的执行过程有四种：

* 嵌套循环法
  * 首先在表1中找到第一个元组，然后逐一查找表2中符合条件的拼接成结果
  * 表2全部查找完后，再找表1中第二个元组，然后再从头开始扫描表2
  * 重复上述操作，直到表1中的全部元组都处理完毕
* 排序合并法
  * 常用于等值（`=`）连接
  * 首先按连接属性对表1和表2排序
  * 对表1的第一个元组，逐一顺序查找表2中满足连接条件的元组，当遇到第一条大于表1连接字段值的元组时中断
  * 找到表1的第二条元组，然后从刚才的中断点处继续顺序扫描表2
* 索引连接法
  * 对表2按连接字段建立索引
  * 对表1中的每个元组，依次根据其连接字段值查询表2的索引，从中找到满足条件的元组

【概念】自然连接：连接是从两个关系的笛卡尔积中选取满足一定条件的元组，它分为自然连接和等值连接。自然连接是等值连接后去掉了相同的记录。

```mysql
# 查询每个学生及其选修课程的情况（自然连接）
SELECT Student.Sno,Sname,Ssex,Sage,Sdept,Cno,Grade # Student.Sno 是关键，清除重复
FROM Student,SC
WHERE Student.Sno = SC.Sno;
```

#### 2.2.2 自身连接

自身连接是一个表与其自己进行连接，因此需要给表起别名以示区别。由于所有属性名都是同名属性，因此必须使用别名前缀。

```mysql
# 查询每一门课的间接先修课（即先修课的先修课）
SELECT FIRST.Cno, SECOND.Cpno
FROM Course FIRST, Course SECOND
WHERE FIRST.Cpno = SECOND.Cno;
```

#### 2.2.3 外连接

外连接与普通连接的区别：

* 普通连接（内连接）操作只输出满足连接条件的元组
* 外连接操作以指定表为连接主体，将主体表中不满足连接条件的元组一并输出
  * 左外连接：列出左边关系中所有的元组
  * 右外连接：列出右边关系中所有的元组

```mysql
# 查询每个学生及其选修课程的情况（左外连接）
SELECT Student.Sno,Sname,Ssex,Sage,Sdept,Cno,Grade
FROM Student LEFT JOIN SC ON (Student.Sno=SC.Sno);
```

数据库在通过连接两张或多张表来返回记录时，都会生成一张**中间的临时表**，然后再将这张临时表返回给用户。 在使用 `left jion` 时，`on` 和 `where` 条件的**区别**如下：

1. `on` 条件是在**生成临时表时**使用的条件，它不管 `on` 中的条件是否为真，都会返回左边表中的记录。

2. `where` 条件是在**临时表生成好后**，再对临时表进行过滤的条件。这时已经没有 `left join` 的含义（必须返回左边表的记录）了，条件不为真的就全部过滤掉。

#### 2.2.4 多表连接

将两个以上的表进行连接：

```mysql
SELECT Student.Sno, Sname, Cname, Grade
FROM Student, SC, Course # 多表连接
WHERE Student.Sno = SC.Sno AND SC.Cno = Course.Cno;
```

### 2.3 嵌套查询

一个`SELECT-FROM-WHERE`语句称为一个查询块，将一个查询块嵌套在另一个查询块的`WHERE`子句
或`HAVING`短语的条件中，这称为嵌套查询。

子查询有一个限制：不能使用`ORDER BY`子句。

求解方法：

* 不相关子查询（子查询的查询条件不依赖于父查询）
  * 由里向外，逐层处理
* 相关子查询（子查询的查询条件依赖于父查询）
  * 首先取外层查询中表的第一个元组，根据它与内层查询相关的属性值处理内层查询，若WHERE子句返回值为真，则取此元组放入结果表
  * 然后再取外层表的下一个元组
  * 重复这一过程，直至外层表全部检查完为止

#### 2.3.1 带有`IN`谓词的子查询

```mysql
# 查找与'刘晨'在同一个系的学生
SELECT Sno, Sname, Sdept # 2 查找所有在 CS 系学习的学生
FROM Student
WHERE Sdept IN
    (SELECT Sdept # 1 确定“刘晨”所在系名，结果为：CS
    FROM Student
    WHERE Sname='刘晨'); # 此查询为不相关子查询

# 用自身连接完成
SELECT S1.Sno, S1.Sname,S1.Sdept
FROM Student S1,Student S2
WHERE S1.Sdept = S2.Sdept AND S2.Sname = '刘晨';
```

#### 2.3.2 带有比较运算符的子查询

当能确切知道内层查询返回单值时，可用比较运算符（`>`，`<`，`=`，`>=`，`<=`，`!=`或`<>`）。

```mysql
# 查找与'刘晨'在同一个系的学生
SELECT Sno, Sname, Sdept # 2 查找所有在 CS 系学习的学生
FROM Student
WHERE Sdept = # 由于一个学生只可能在一个系学习，则可以用 = 代替 IN
    (SELECT Sdept # 1 确定“刘晨”所在系名，结果为：CS
     FROM Student
     WHERE Sname='刘晨'); # 此查询为不相关子查询

# 选取每个人大于自己平均成绩的课程成绩
SELECT Sno, Cno
FROM SC x
WHERE Grade>=
	(SELECT AVG(Grade)
     FROM SC y
     WHERE y.Sno=x.Sno); # 此查询为相关子查询
     
/*
过程分析：

从外层查询中取出SC的一个元组x，将元组x的Sno值（201215121）传送给内层查询：

    SELECT AVG(Grade)
    FROM SC y
    WHERE y.Sno='201215121‘;

执行内层查询，得到值88（近似值），用该值代替内层查询，得到外层查询：

    SELECT Sno,Cno
    FROM SC x
    WHERE Grade >=88;

执行这个查询，得到：

    (201215121,1)
    (201215121,3)

然后外层查询取出下一个元组重复做上述①至③步骤，直到外层的SC元组全部处理完毕。结果为：

    (201215121,1)
    (201215121,3)
    (201215122,2)
*/
```

#### 2.3.3 带有`ANY(SOME)`或`ALL`谓词的子查询

使用`ANY`或`ALL`谓词时必须同时使用比较运算符。

```mysql
SELECT Sname,Sage
FROM Student
WHERE Sage<ANY
    (SELECT Sage
     FROM Student
     WHERE Sdept='CS')
AND Sdept<>'CS';

# 用聚集函数实现
SELECT Sname,Sage
FROM Student
WHERE Sage<
    (SELECT MAX(Sage) # 聚集函数
     FROM Student
     WHERE Sdept='CS')
AND Sdept<>'CS';
```

`ANY`（或`SOME`），`ALL`谓词与聚集函数、`IN`谓词的等价转换关系：

|       | `=`  | `<>`或`!=` | `<`    | `<=`    | `>`    | `>=`    |
| ----- | ---- | ---------- | ------ | ------- | ------ | ------- |
| `ANY` | `IN` | -          | `<MAX` | `<=MAX` | `>MIN` | `>=MIN` |
| `ALL` | -    | `NOT IN`   | `<MIN` | `<=MIN` | `>MAX` | `>=MAX` |

#### 2.3.4 带有`EXISTS`谓词的子查询

`EXISTS`谓词是存在量词，带有`EXISTS`谓词的子查询不返回任何数据，只产生逻辑真值`true`或逻辑假值`false`。

* 若内层查询结果非空，则外层的`WHERE`子句返回真值
* 若内层查询结果为空，则外层的`WHERE`子句返回假值

`NOT EXISTS`谓词反之。

由`EXISTS`引出的子查询，其目标列表达式通常都用`* `；因为带`EXISTS`的子查询只返回真值或假值，给出列名无实际意义。

```mysql
# 查询没有选修1号课程的学生姓名
SELECT Sname
FROM Student
WHERE NOT EXISTS
    (SELECT *
     FROM SC
     WHERE Sno = Student.Sno AND Cno='1');
```

难点：用`EXISTS/NOT EXISTS`实现逻辑蕴涵

* `SQL`语言中没有蕴涵（`Implication`）逻辑运算

* 可以利用谓词演算将逻辑蕴涵谓词等价转换为：$p \to q \Leftrightarrow \overline{p} \vee q \Leftrightarrow \overline{p \wedge \overline{q}}$

  > 在我们平时遇到的“p→q”要么是p与q之间存在着一定的联系，要么是在前件p为真的前提下作出的判断。就比如我们在高中时学过的“命题”，它的形式是“若p，则q”，形式与“蕴含式”一模一样，但是细心的同学就会发现，我们在研究问题的时候通常将前件p视为真命题，然后再来研究后件q的真假性，从而判断“命题”真假性。所以说我们用以前高中的的思路是很难理解这个真值表（True Table）的。
  > 　　
  >  但是要是真想理解起来倒也不是很难，举个栗子：这里存在着一个蕴含式“如果我当上了班长，我将提高同学的福利”。所以不妨设，前件p为“我当上了班长”，后件q为“我将提高同学的福利”，那么这个蕴含式就可以符号化为p→q，记为 I。
  >
  >  所以问题就是判断I的真假，也就是说我有没有撒谎。
  >
  >  很显然当p为真，q为假的时候，我并没有履行我的诺言，故此时 I 为"False"。当p为假的时候，不管我有没有提高同学们的福利，你都不能说我没有履行我的诺言，因为我就没有当上班长，此时的I为真。

```mysql
# 查询至少选修了学生201215122选修的全部课程的学生号码

/*
用逻辑蕴涵表达：查询学号为x的学生，对所有的课程y，只要201215122学生选修了课程y，则x也选修了y。

用P表示谓词 “学生201215122选修了课程y”
用q表示谓词 “学生x选修了课程y”
则上述查询为: （任意y） p 推 q，可以转换为 与、或、非 的逻辑表达，这样就能用 EXIST 实现
*/
SELECT DISTINCT Sno
FROM SC SCX
WHERE NOT EXISTS
    (SELECT *
     FROM SC SCY
     WHERE SCY.Sno = '201215122' AND NOT EXISTS
         (SELECT *
          FROM SC SCZ
          WHERE SCZ.Sno=SCX.Sno AND SCZ.Cno=SCY.Cno));
```

### 2.4 集合查询

集合操作的种类：

* 并操作`UNION`
* 交操作`INTERSECT`
* 差操作`EXCEPT`

**【注】**参加集合操作的各查询结果的列数必须相同；对应项的数据类型也必须相同。

```mysql
SELECT *
FROM Student
WHERE Sdept='CS'
UNION
SELECT *
FROM Student
WHERE Sage<=19;
```

### 2.5 基于派生表的查询

子查询不仅可以出现在`WHERE`子句中，还可以出现在`FROM`子句中，这时子查询生成的临时派生表（Derived Table）成为主查询的查询对象。

```mysql
# 找出每个学生超过他自己选修课程平均成绩的课程号
SELECT Sno, Cno
FROM SC, (SELECT Sno, Avg(Grade)
          FROM SC
          GROUP BY Sno)
          AS Avg_sc(avg_sno,avg_grade) # 有聚集函数，因此表名称和列名称同时起别名
WHERE SC.Sno = Avg_sc.avg_sno and SC.Grade >= Avg_sc.avg_grade

SELECT Sname
FROM Student,
	(SELECT Sno FROM SC WHERE Cno=' 1 ') AS SC1 # 无聚集函数，可以不指定列名称（属性列）
WHERE Student.Sno=SC1.Sno;
```

### 2.6 查询技巧

#### 2.6.1 设计返回 null

设计好满足条件的查询后，为了返回 `null`，再在外面套一层即可。

```mysql
# 获取 Employee 表中第二高的薪水（Salary）
select
(
    select Salary
    from Employee e1
    where 1=
        (
            select count(*)
            from Employee e2
            where e2.Salary > e1.Salary
        )
)
as SecondHighestSalary
;

# 使用 LIMIT 的另一种解法
select (select distinct salary from Employee order by salary desc limit 1,1) as SecondHighestSalary;
```

#### 2.6.2 排名问题

方法一：使用函数

```mysql
# 排名不连续
select
    Score,
    rank() over (
        order by Score desc
    ) "Rank"
from Scores;

# 排名连续
select
    Score,
    dense_rank() over (
        order by Score desc
    ) "Rank"
from Scores;
```

方法二：统计

```mysql
select
	a.Score as Score, # 1) 对每个降序排序的分数
	(
        select count(distinct b.Score) # distinct 保证所有相同的分数只计算 1 次，得到连续的排名
        from Scores b
        where b.Score >= a.Score # 2) 统计比它大的分数个数 # 等号保证排名从 1 开始
    ) as "Rank"
from Scores a
order by a.Score DESC
```

参考题目：

* lc 180. 连续出现的数字



## 3 数据更新

### 3.1 插入数据

#### 3.3.1 插入元组

```mysql
INSERT
INTO <表名> [(<属性列1> [,<属性列2>...)]
VALUES (<常量1> [,<常量2>]... );
```

关于`INTO`子句：

* 指定要插入数据的表名及属性列，属性列的顺序可与表定义中的顺序不一致
* 没有指定属性列：表示要插入的是一条完整的元组，且属性列属性与表定义中的顺序一致
* 指定部分属性列：插入的元组在其余属性列上取空值

关于`VALUES`子句：

* 提供的值必须与`INTO`子句匹配（包括值的个数和值的类型）

```mysql
# 将一个新学生元组（学号：201215128；姓名：陈冬；性别：男；所在系：IS；年龄：18岁）插入到 Student 表中
INSERT
INTO Student(Sno,Sname,Ssex,Sdept,Sage)
VALUES ('201215128','陈冬','男','IS',18);
```

#### 3.3.2 插入子查询结果

```mysql
INSERT
INTO <表名> [(<属性列1> [,<属性列2>… )]
子查询;
```

关于子查询：

* `SELECT`子句目标列必须与`INTO`子句匹配（包括值的个数和值的类型）

```mysql
# 对每一个系，求学生的平均年龄，并把结果存入数据库

# 第一步：建表
CREATE TABLE Dept_age(
    Sdept CHAR(15)
	Avg_age SMALLINT
);
# 第二步：插入数据
INSERT
INTO Dept_age(Sdept,Avg_age)
    SELECT Sdept，AVG(Sage)
    FROM Student
    GROUP BY Sdept;
```

### 3.2 修改数据

```mysql
UPDATE <表名>
SET <列名>=<表达式>[,<列名>=<表达式>]…
[WHERE <条件>];
```

* 修改指定表中满足`WHERE`子句条件的元组
* `SET`子句给出`<表达式>`的值用于取代相应的属性列
* 如果省略`WHERE`子句，表示要修改表中的所有元组形式

#### 3.2.1 修改某一个元组的值

```mysql
# 将学生201215121的年龄改为22岁
UPDATE Student
SET Sage=22
WHERE Sno='201215121';
```

#### 3.2.2 修改多个元组的值

```mysql
# 将所有学生的年龄增加1岁
UPDATE Student
SET Sage=Sage+1;
```

#### 3.2.3 带子查询的修改语句

```mysql
# 将计算机科学系全体学生的成绩置零
UPDATE SC
SET Grade=0
WHERE Sno IN
    (SELETE Sno
     FROM Student
     WHERE Sdept= 'CS' );
```

### 3.3 删除数据

```mysql
DELETE
FROM <表名>
[WHERE <条件>];
```

* 删除指定表中满足`WHERE`子句条件的元组
* `WHERE`子句指定要删除的元组
* `WHERE`子句缺省表示要删除表中的全部元组，表的定义仍在字典中

#### 3.3.1 删除某一个元组的值

```mysql
# 删除学号为201215128的学生记录
DELETE
FROM Student
WHERE Sno= 201215128 ';
```

#### 3.3.2 删除多个元组的值

```mysql
# 删除所有的学生选课记录
DELETE
FROM SC;
```

#### 3.3.3 删除计算机科学系所有学生的选课记录

```mysql
DELETE
FROM SC
WHERE Sno IN
    (SELECT Sno
     FROM Student
     WHERE Sdept='CS');
```

### 3.4 完整性检查

#### 3.4.1 插入检查

关系数据库管理系统在执行插入语句时会检查所插元组是否破坏表上已定义的完整性规则：

* 实体完整性
* 参照完整性
* 用户定义的完整性
  * `NOT NULL`约束
  * `UNIQUE`约束
  * 值域约束

#### 3.4.2 修改检查

关系数据库管理系统在执行修改语句时会检查修改操作是否破坏表上已定义的完整性规则：

* 实体完整性
* 主码不允许修改
* 用户定义的完整性
  * `NOT NULL`约束
  * `UNIQUE`约束
  * 值域约束

## 4 空值处理

空值是一个很特殊的值，含有不确定性。对关系运算带来特殊的问题，需要做特殊的处理。

### 4.1 空值的判断

判断一个属性的值是否为空值，用`IS NULL`或`IS NOT NULL`来表示。

### 4.2 空值的约束条件

属性定义（或者域定义）中

* 有`NOT NULL`约束条件的不能取空值
* 加了`UNIQUE`限制的属性不能取空值
* 码属性不能取空值

### 4.3 空值的运算

空值与另一个值（包括另一个空值）的算术运算的结果为空值。

空值与另一个值（包括另一个空值）的比较运算的结果为`UNKNOWN` 。

有`UNKNOWN`后，传统二值（`TRUE`，`FALSE`）逻辑就扩展成了三值逻辑，运算规则为能出结果就出结果，否则为`UNKNOWN`。

