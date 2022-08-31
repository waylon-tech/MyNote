## 目录

[toc]

## 1 窗口函数

### 1.1 排序窗口函数

#### `row_number`

**含义**

```hive
ROW_NUMBER() OVER(PARTITION BY COLUMN1 ORDER BY COLUMN2 DESC)
```

首先根据 `COLUMN1` 进行结果集分组，结果集内部按照 `COLUMN2` 排序，输出结果是类似于**双重分组**的结果。

<u>例1：每个部门的员工按照工资降序排序</u>

```hive
select
    *,row_number() over(partition by dept order by salary desc) as rn
from
    ods_num_window
;
```

<img src="img/hive技巧_row_number例.jpg" alt="hive技巧_row_number例" style="zoom: 80%;" />

<u>例2：全部的员工按照工资降序排序</u>

```hive
select
    *,row_number() over(order by salary desc) as rn
from
    ods_num_window
;
```

**场景**

> Top-N 查询
>
> 根据某个值排序，取其中 Top-N 的行。

<u>例：取每个部门的工资前两名</u>

在每个部门内按照薪资排序（双重分组）后，取排序为 Top-N 的行即可。

```hive
select *
from(
   select
       *,row_number() over(partition by dept order by salary desc) as rn
   from
       ods_num_window
) tmp
where
    rn <=2
;
```

<img src="img/hive技巧_row_number_topn查询.jpg" alt="hive技巧_row_number_topn查询" style="zoom:80%;" />

> 计算连续

<u>例：计算连续访问天数最大的 10 位用户</u>

针对同一个用户，按照其访问时间排序（双重分组），然后我们用日期的数字减去对应的排序得到一个值，如果访问时间是连续的话，这个值就一定相同，从而能够判断连续性。

```hive
-- 1 查验数据
select
    id,ctime,
    row_number(partition by id order by ctime ) as rn
from
    ods_user_log
;
```

```hive
-- 2 日期减法
select
    id, ctime,
    date_sub(cast(ctime as date), row_number() over(partition by id order by ctime)),
    row_number() over(partition by id order by ctime ) as rn
from
    ods_user_log
;
```

<img src="img/hive技巧_row_number_连续查询.jpg" alt="hive技巧_row_number_连续查询" style="zoom: 65%;" />

```hive
-- 3 统计连续
select
    id,kt,count(1) as loginCnt
from (
    select
        id,ctime,
        date_sub(cast(ctime as date),row_number() over(partition by id order by ctime)) as kt,
        row_number() over(partition by id order by ctime ) as rn
    from
        ods_user_log
) tmp
group by
    id,kt
having
    count(1)>=7
;
```

> 分组抽样

https://zhuanlan.zhihu.com/p/342682617 未完待续 0831 -> 0901