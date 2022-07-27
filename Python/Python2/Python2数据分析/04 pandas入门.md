###4-1 pandas的安装与概览
* A:pandas
	* pandas：数据分析程序库（本章主要学习类关系型数据库的处理手段）
	* 最小的依赖项集合
		* NumPy：这是一个处理数值数组的基础软件包，前文已经介绍
		* python-dateutil：这是一个专门用来处理日期数据的程序库
		* pytz：这是一个处理时区问题的程序库

* B:安装
* 
		pip install pandas

* C:研究pandas的相关文档
	* 
			import pandas as pd
			import pkgutil as pu
			import pydoc

			def print_desc(prefix, pkg_path):
			    for pkg in pu.iter_modules(path=pkg_path):
			        name = prefix + "." + pkg[1]
			
			        if pkg[2] == True:
			            try:
			                docstr = pydoc.plain(pydoc.render_doc(name))
			                docstr = clean(docstr)
			                start = docstr = docstr[start: start + 140]
			                print name, docstr
			            except:
			                continue
			def clean(astr):
			s = astr
			#remove multiple spaces
			s = ' ',join(s.split())
			s = s.replace('=', ' ')
			return s
			
			print_desc("pandas", pd.__path__)
			print "end"

			Out:
			#None!

###4-2 pandas数据结构之DataFrame
* A:介绍
	* pandas的DataFrame数据结构是一种带标签的二维对象，与Excel的电子表格或者关系型数据库的数据表非常类似
	* DataFrame的概念最初源于R语言

* B:DataFrame的创建方式
	* 1 从另一个DataFrame创建DataFrame
		* 复制
	* 2 从具有二维形状的NumPy数组或者数组的复合结构来生成DataFrame
		* DataFrame([[],[]], index=) - 自动选取自然数序列为index，也可人为指定
		* 例如：
			* 用字典创建
			* 
					pandas.DataFrame({'title' : list, 'title' :list, ... ...})

			* 列标签会按照Python字典的各个健的词汇顺序进行创建
	* 3 用pandas的另外一种数据结构Series来创建DataFrame
		* 
	* 4 从类似CSV之类的文件来生成
		* pandas.io.read_csv("path") - 读取CSV文件，返回DataFrame数据类型

* C:DataFrame的属性
	* 案例演示
		* 将数据载入DataFrame，并显示其内容
		* 
				from pandas.io.parsers import read_csv
	
				df = read_csv("UNdata.csv")
				print "DataFrame", df
	
				Out:
					DataFrame          Country or Area  Year          Value
						0                    Australia  2014   98076.109401
						1                    Australia  2013   99857.204626
					#more ... ... ...
						[1074 rows x 3 columns]

	* 对DataFrame整体：
		* shape - 返回DataFrame形状数据的元组
		* len() - 总数据条数
		* 
				print "Shape", df.shape
				print "Length", len(df)
	
				Out:
					Shape (1074, 3)
					Length 1074

	* 对title标题：
		* columns - 用一个专门的数据结构来容纳标题 注：可加[]进行切片选择，得到的是标题字符串，结合下面的DataFrame["title"]使用
		* dtypes - 标题的数据类型
		* 
				print "Column Header\n", df.columns
				print "Data types\n", df.dtypes
	
				Out:
					Column Header
					Index([u'Country or Area', u'Year', u'Value'], dtype='object')
					Data types
					Country or Area     object
					Year                 int64
					Value              float64
					dtype: object

	* 对每一条数据：
		* index - 索引，类似于关系型数据库中的数据表的主键（primary key）
		* 
				print "Index", df.index
	
				Out:
					Index RangeIndex(start=0, stop=1074, step=1)
	
		* values - 得到所有的数值列表；注意，非数值部分被表位nan
		* 
				print "Values", df.values
	
				Out:
					Values [['Australia' 2014L 98076.1094012298]
					 ['Australia' 2013L 99857.20462570449]
					 ['Australia' 2012L 100796.838744057]
					 ..., 
					 ['United States of America' 1992L 776869.789751931]
					 ['United States of America' 1991L 777034.2209150121]
					 ['United States of America' 1990L 773854.896420295]]

###4-3 pandas数据结构之Series
* A:介绍
	* pandas的Series数据结构是由不同类型的元素组成的一维数组，该数据结构也具有标签

* B:Series的创建方式
	* 1 由Python的字典来创建
		* pandas.Series(dict)
	* 2 由NumPy数组来创建Series
		* pandas.Series(array[])
	* 3 由单个标量值来创建Series
		* pandas.Series(lists, lists)
	* 4 从DataFrames中提取
		* dataframe["title"] - 选中title列
		* 案例演示
		* 
				from pandas.io.parsers import read_csv
	
				df = read_csv("UNdata.csv")
				print "DataFrame", df
				
				country_col = df["Country or Area"]
				print "Type df", type(df)
				print "Type country col", type(country_col)
	
				Out:
					[1074 rows x 3 columns]
					Type df <class 'pandas.core.frame.DataFrame'>
					Type country col <class 'pandas.core.series.Series'>

	* 关于Series构造函数的可选参数：轴标签（通常称为索引）
		* 默认情况，如使用NumPy数组作为输入数据，那么pandas会将索引值从0开始自动递增
		* 如果一个Python字典dict，dict的健会经排序后变成相应的索引
		* 如果输入数据是一个标量值，就需要人为逐个提供相应的索引

* C:Series的属性
	* shape - 以元组的形式来存放DataFrame形状数据
	* name - 获得标题名称
	* index - 索引，类似于关系型数据库中的数据表的主键（primary key）
	* values - 得到所有的数值列表；注意，非数值部分被表位nan
	* 
			print "Series shape", country_col.shape
			print "Series name", country_col.name
			print "Series index", country_col.index
			print "Series values", country_col.values

			Out:
				Series shape (1074L,)
				Series name Country or Area
				Series index RangeIndex(start=0, stop=1074, step=1)
				Series values ['Australia' 'Australia' 'Australia' ..., 'United States of America'
				 'United States of America' 'United States of America']

* D:DataFrame和Series的共同功能
	* dataframe或series[#切片语法] - seies的切片功能（同下查询功能）
	* 
			print "Last 2 countries\n", country_col[-2:]
			print "Last 2 countries", type(country_col[-2:1])

			Out:
				Last 2 countries
				1072    United States of America
				1073    United States of America
				Name: Country or Area, dtype: object
				Last 2 countries <class 'pandas.core.series.Series'>

	* NumPy模块的兼容类型
		* numpy.sign() - NumPy的函数同样适用于pandas的DataFrame和Series数据结构，该函数的作用是返回数字的符号0
		* 
				print "df sign\n", np.sign(df)
				last_col = df.columns[-1]
				print "Last df column signs\n", last_col, np.sign(df[last_col])
	
				Out:
					df sign
					     Country or Area Year Value
						0                  1    1     1
						1                  1    1     1
						#略
						[1074 rows x 3 columns]
	
						Last df column signs
						Value 0       1.0
						1       1.0
						2       1.0
						#略
						Name: Value, Length: 1074, dtype: float64


	* dataframe减series可以得到的两种结果及方法：
		* 一个数组，各个元素以0填充，并且至少有一个NaN（对NumPy函数来说，大部分涉及NaN的运算都会生成NaN）
		* 
				print sum([0, np.nan])
				Out:
					nan

		* 一个元素全为0的数组
		* 

				print np.sum(df[last_col] - df[last_col].values)
				Out:
					0.0

###4-4 利用pandas查询数据
* A:案例演示
	* 从Qualdl检索年度太阳黑子数据，可以使用Qualdl API，也可以从网站（http://www.quandl.com/SIDC/SUNSPOTS_A-Sunspot-Numbers-Annual）下载CSV文件
	* 对于第一种方法：
	* 
			pip install Quandl

			import Quandl

			# Data from http://www.quandl.com/SIDC/SUNSPOTS_A-Sunspot-Numbers-Annual
			#Pypi url https://pypi.python.org/pypi/Quandl
			suspots = Quandl.get("SIDC/SUNSPOTS_A")

* B:查询函数
	* dataframe.head(n) 与 dataframe.tail(n) - 选取DataFrame的前n和后n个数据记录
	* 
			from pandas.io.parsers import read_csv

			ag = read_csv("UNdata_Agriculture, value added (% of GDP).csv")
			print "Head 2", ag.head(2)
			print "Tail 2", ag.tail(2)

			Out:
				Head 2   Country or Area  Year      Value  Value Footnotes
				0     Afghanistan  2015  22.603927              NaN
				1     Afghanistan  2014  23.463126              NaN
				Tail 2      Country or Area  Year      Value  Value Footnotes
				7564        Zimbabwe  1966  21.953762              NaN
				7565        Zimbabwe  1965  20.125818              NaN

	* dataframe.loc[a:b,['t1','tn']] - a:b表示起止行（含头不含尾），['t1','tn']表示要提取的标签的列表
	* dataframe.iloc[a:b,[1, n]] - a:b表示起止行（含头不含尾），[1, n]表示要提取标签的位置的列表
	* dataframe.at[a:b,['t1','tn']] -  a:b表示起止行（含头不含尾），['t1','tn']表示要提取的标签的列表
	* dataframe.iat[index] - 直接根据index（坐标）返回结果
	* 注：a:b可为单个数，['','']或[,]可省略
	* 
			print ag.loc[1:10,["Year"]]
			last_date = ag.index[-5]

			print "Last value\n", ag.loc[last_date]

			Out:
				    Year
				1   2014
				2   2013
				3   2012
				4   2011
				5   2010
				6   2009
				7   2008
				8   2007
				9   2006
				10  2005
				Last value
				Country or Area    Zimbabwe
				Year                   1965
				Value               20.1258
				Value Footnotes         NaN
				Name: 7565, dtype: object

			print ag.iloc[1:10, [1,2]]
			print "Scalar with iloc", ag.iloc[0,0]
			print "Scalar with iat", ag.iat[0,1]

			Out:
				   Year      Value
				1  2014  23.463126
				2  2013  23.891372
				3  2012  24.603247
				4  2011  24.507440
				5  2010  27.091540
				6  2009  30.205602
				7  2008  25.394741
				8  2007  30.622854
				9  2006  29.249737
				Scalar with iloc Afghanistan
				Scalar with iat 2015

	* dataframe[#切片语法] - 切片查询
	* 
			print "Values slice by date", ag[2010: 2015]
			Out:
				Values slice by date      Country or Area  Year      Value  Value Footnotes
				2010         Ecuador  2011   9.944893              NaN
				2011         Ecuador  2010  10.180392              NaN
				2012         Ecuador  2009  10.503001              NaN
				2013         Ecuador  2008   9.298670              NaN
				2014         Ecuador  2007   9.837195              NaN

	* dataframe[判断表达式] - 查询布尔型变量
	* 
			#不符合的赋NaN值
			print "Boolean selection\n", ag[ag > ag.mean()]
			#不符合的不返回
			print "Boolean selection with column label\n", ag[ag.Value > ag.Value.mean()]

			Out:
				Boolean selection
				     Country or Area    Year      Value  Value Footnotes
				0        Afghanistan  2015.0  22.603927              NaN
				1        Afghanistan  2014.0  23.463126              NaN
				#略
				[7566 rows x 4 columns]
				Boolean selection with column label
				     Country or Area  Year      Value  Value Footnotes
				0        Afghanistan  2015  22.603927              NaN
				1        Afghanistan  2014  23.463126              NaN
				#略
				
				[3248 rows x 4 columns]

###4-5 利用pandas的DataFrame进行统计计算
* A:pandas的统计函数
	* describe() 	- 这个方法将返回描述性统计信息
	* count() 		- 这个方法将返回非NaN数据项的数量
	* mad() 		- 这个方法用于计算平均绝对偏差（mean absolute deviation），即类似于标准差的一个有力统计工具
	* median()		- 这个方法将返回中位数，等价于第50百分位数的值
	* min() 		- 这个方法返回最小值
	* max() 		- 这个方法返回最大值
	* mode() 		- 这个方法将返回众数（mode），即一组数据中出现次数最多的变量值
	* std() 		- 这个方法将返回表示离散度（disperation）的标准差，及方差的平方根
	* var() 		- 这个方法将返回方差
	* skew() 		- 这个方法用来返回偏态函数（skewness），该系数表示的是数据分布的对称程度
	* kurt() 		- 这个方法返回峰态系数（kurtosis），该系数用来反映数据分布曲线顶端尖峭或扁平程度

* B:案例演示
	* 
			from pandas.io.parsers import read_csv

			df = read_csv("UNdata_Agriculture, value added (% of GDP).csv")
			print "Describe", df.describe()
			print "Non NaN observation", df.count()
			print "MAD", df.mad()
			print "Median", df.median()
			print "Min", df.min()
			print "Max", df.max()
			print "Mode", df.mode()
			print "Standar", df.std()
			print "Variance", df.var()
			print "Skewness", df.skew()
			print "Kurtosis", df.kurt()

			Out:
				Describe               Year        Value  Value Footnotes
				count  7566.000000  7566.000000        27.000000
				mean   1993.821042    19.039032         2.037037
				std      14.339983    15.667656         0.192450
				min    1960.000000     0.000000         2.000000
				25%    1984.000000     5.749572         2.000000
				50%    1996.000000    15.084892         2.000000
				75%    2006.000000    29.384718         2.000000
				max    2015.000000    94.846403         3.000000
				Non NaN observation Country or Area    7566
				Year               7566
				Value              7566
				Value Footnotes      27
				dtype: int64
				MAD Year               12.006554
				Value              12.924564
				Value Footnotes     0.071331
				dtype: float64
				Median Year               1996.000000
				Value                15.084892
				Value Footnotes       2.000000
				dtype: float64
				Min Country or Area    Afghanistan
				Year                      1960
				Value                        0
				Value Footnotes              2
				dtype: object
				Max Country or Area    Zimbabwe
				Year                   2015
				Value               94.8464
				Value Footnotes           3
				dtype: object
				Mode                           Country or Area    Year  Value  Value Footnotes
				0                              Bangladesh  2006.0    0.0              2.0
				1                                   Benin  2007.0    NaN              NaN
				2                                  Brazil     NaN    NaN              NaN
				3                            Burkina Faso     NaN    NaN              NaN
				4                                    Chad     NaN    NaN              NaN
				5                                   Chile     NaN    NaN              NaN
				6                                   China     NaN    NaN              NaN
				7                             Congo, Rep.     NaN    NaN              NaN
				8                           Cote d'Ivoire     NaN    NaN              NaN
				9   East Asia & Pacific (developing only)     NaN    NaN              NaN
				10                                Ecuador     NaN    NaN              NaN
				11                                 Guyana     NaN    NaN              NaN
				12                               Honduras     NaN    NaN              NaN
				13                              Indonesia     NaN    NaN              NaN
				14                                  Kenya     NaN    NaN              NaN
				15                    Low & middle income     NaN    NaN              NaN
				16                                 Malawi     NaN    NaN              NaN
				17                               Malaysia     NaN    NaN              NaN
				18                          Middle income     NaN    NaN              NaN
				19                                  Niger     NaN    NaN              NaN
				20                               Pakistan     NaN    NaN              NaN
				21                            Philippines     NaN    NaN              NaN
				22                           South Africa     NaN    NaN              NaN
				23                              Sri Lanka     NaN    NaN              NaN
				24                                  Sudan     NaN    NaN              NaN
				25                              Swaziland     NaN    NaN              NaN
				26                                   Togo     NaN    NaN              NaN
				27                                 Turkey     NaN    NaN              NaN
				28                                 Uganda     NaN    NaN              NaN
				29                    Upper middle income     NaN    NaN              NaN
				Standar Series([], dtype: float64)
				Variance Series([], dtype: float64)
				Skewness Series([], dtype: float64)
				Kurtosis Series([], dtype: float64)

###4-6 利用pandas的DataFrame实现数据聚合
* A:数据聚合概念
	* 数据聚合是关系型数据库中比较常用的一个术语
	* 使用方法 利用查询操作对各行各列中的数据进行分组，然后进行操作
	* pandas的数据结构DataFrame有类似功能：将生成的数据保存到Python字典中，利用这些数据来创建一个pandasDataFrame

* B:pandas的聚合功能
	* a:案例演示
		* 生成一组数据进行聚合
		* 该数据有4列：
			* Weather （一个字符串）
			* Food （一个字符串）
			* Price （一个随机浮点数）
			* Number （1~9之间的一个随机数）
		* 
				import pandas as pd
				from numpy.random import seed
				from numpy.random import rand
				from numpy.random import random_integers
				import numpy as np

				#为NumPy的随机数生成器指定种子，一确保重复运行程序时生成的数据不会走样
				seed(42)
				
				df = pd.DataFrame({'Weather' : ['cold', 'hot', 'cold', 'hot', 'cold',
				                                'hot', 'cold'],
				                    'Food' : ['soup', 'soup', 'icecream', 'chocolate',
				                              'icecream', 'icecream', 'soup'],
				                    'price' : 10 * rand(7),
				                    'Number' : random_integers(1, 9, size=(7,))})
				print df
		
				Out:
					        Food  Number Weather     price
					0       soup       8    cold  3.745401
					1       soup       5     hot  9.507143
					2   icecream       4    cold  7.319939
					3  chocolate       8     hot  5.986585
					4   icecream       8    cold  1.560186
					5   icecream       3     hot  1.559945
					6       soup       6    cold  0.580836

	* b:步骤一：数据分组
		* 数据分组与遍历
		* 
				weather_group = df.groupby('Weather')
				i = 0
				for name, group in weather_group:
				    i = i + 1
				    print "Group", i, name
				    print group
	
				Out:
					       Food  Number Weather     price
					0      soup       8    cold  3.745401
					2  icecream       4    cold  7.319939
					4  icecream       8    cold  1.560186
					6      soup       6    cold  0.580836
					Group 2 hot
					        Food  Number Weather     price
					1       soup       5     hot  9.507143
					3  chocolate       8     hot  5.986585
					5   icecream       3     hot  1.559945

		* dataframe/serie.groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, **kwargs) - 对pandasDataFrame按by进行分组，by可以是mapping, function, str, 或 iterable:
			* 对于字典或Series：会根据值进行分组（A-Z分组）
			* 对于numpy列表，会根据作为值的元素进行分组
			* 对于列表或字符串，会根据DataFrame里相应的列进行分组
			* 对于函数：会传入每一个对象目录的值。返回分组后的多个DataFrame数据组。

		* 了解分组信息
		* 
				wf_group = df.groupby(['Weather', 'Food'])
				print "WF Groups", wf_group.groups
	
				Out:
					WF Groups {('hot', 'chocolate'): Int64Index([3], dtype='int64'), 
								('cold', 'icecream'): Int64Index([2, 4], dtype='int64'),
								('cold', 'soup'): Int64Index([0, 6], dtype='int64'), 
								('hot', 'soup'): Int64Index([1], dtype='int64'),
								('hot', 'icecream'): Int64Index([5], dtype='int64')}

		* group.groups - 返回每一种数据组合的索引值、数据类型，每个数据项都可以通过索引值引用

	* c:步骤二：数据聚合
		* group对象的聚合函数
		* 
				print "Weather group first\n", weather_group.first()
				print "Weather group last\n", weather_group.last()
				print "Weather group mean\n", weather_group.mean()

				Out:
					Weather group first
					         Food  Number     price
					Weather                        
					cold     soup       8  3.745401
					hot      soup       5  9.507143
					Weather group last
					             Food  Number     price
					Weather                            
					cold         soup       6  0.580836
					hot      icecream       3  1.559945
					Weather group mean
					           Number     price
					Weather                    
					cold     6.500000  3.301591
					hot      5.333333  5.684558

		* group.first() - 返回各自数据中的第一行内容
		* group.last() - 返回各自数据中的最后一行内容
		* group.mean() - 返回各组数据的平均值

		* 使用NumPy函数进行聚合
		* 
				print "WF Aggregated\n", wf_group.agg([np.mean, np.median])

				Out:
					WF Aggregated
					                  Number            price          
					                    mean median      mean    median
					Weather Food                                       
					cold    icecream       6      6  4.440063  4.440063
					        soup           7      7  2.163119  2.163119
					hot     chocolate      8      8  5.986585  5.986585
					        icecream       3      3  1.559945  1.559945
					        soup           5      5  9.507143  9.507143

		* group.aggregate(self, arg, *args, **kwargs) - 或者简写agg()，参数arg可以使函数或者Dict
			* 如果是函数：必须是对DataFrame起作用和对DataFrame.apply起作用
			* 如果是Dict：键值必须是DataFrame的列名。
				* 可以接受的组合有：
					* string cythonized function name
					* 函数或函数列表（NumPy函数式特殊情形，有默认行为）
					* 列-函数Dict
					* nested dict of names -> dicts of functions
			* 返回聚合后的DataFrame

###4-7 DataFramed的串联与附加操作
* A:介绍
	* 数据库的数据表有内部连接和外部连接两种操作类型。
	* pandas也有类似操作

* B:数据的串联与附加操作
	* 
			print "df: 3\n", df[:3]

			Out:
				df: 3
				       Food  Number Weather     price
				0      soup       8    cold  3.745401
				1      soup       5     hot  9.507143
				2  icecream       4    cold  7.319939

	* pandas.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True) - 串联DataFrame，objs为一个Series的sequence或mapping, DataFrame,或Panel objects，如果是一个字典，分类后的键将会传入。
	* 
			print "Concat Back together\n", pd.concat([df[:3], df[3:]])

			Out:
				Concat Back together
				        Food  Number Weather     price
				0       soup       8    cold  3.745401
				1       soup       5     hot  9.507143
				2   icecream       4    cold  7.319939
				3  chocolate       8     hot  5.986585
				4   icecream       8    cold  1.560186
				5   icecream       3     hot  1.559945
				6       soup       6    cold  0.580836

	* 更多实例请看help()

	* dataframe.append(self, other, ignore_index=False, verify_integrity=False) - 将数据ohter追加到dataframe末尾
	* 
			print "Appending rows\n", df[:3].append(df[5:])

			Out:
				Appending rows
				       Food  Number Weather     price
				0      soup       8    cold  3.745401
				1      soup       5     hot  9.507143
				2  icecream       4    cold  7.319939
				5  icecream       3     hot  1.559945
				6      soup       6    cold  0.580836

	* 更多实例请看help()

###4-8 连接DataFrames
* A:介绍
	* pandas支持所有的连接类型，这里仅介绍内部链接与完全外部连接
	* 对于内部连接：它将从两个数据表中选取数据，只要两个表中连接条件规定的列上存在匹配值，相应的数据就会被组合起来。
	* 对于外部链接：由于不要求进行匹配处理，所以将返回更多的数据

* B:内部连接与外部连接
	* 
			import pandas as pd

			dest = pd.DataFrame({'EmpNr' : [5, 3, 9],
			                    'Dest' : ['Hague', 'Amsterdam', 'Rotterdam']})
			print dest
			
			tips = pd.DataFrame({'EmpNr' : [5, 9, 7],
			                     'Amount' : [10, 5, 2.5]
			    })
			print tips

			Out:
				        Dest  EmpNr
				0      Hague      5
				1  Amsterdam      3
				2  Rotterdam      9
				   Amount  EmpNr
				0    10.0      5
				1     5.0      9
				2     2.5      7

	* pandas.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False) - 连接left和right数据库
		* left_index - 用left数据的目录进行连接
		* right_index - 用right数据的目录进行连接
	* dataframe.join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False) - 连接other到目标数据库
	* 
			print "Merge() on key\n", pd.merge(dest, tips, on='EmpNr')	

			Out:
				        Dest  EmpNr  Amount
				0      Hague      5    10.0
				1  Rotterdam      9     5.0

			print "Dest join() tips\n", dest.join(tips, lsuffix='Dest', rsuffix='Tips')

			Out:
					Dest  EmpNrDest  Amount  EmpNrTips
			0      Hague          5    10.0          5
			1  Amsterdam          3     5.0          9
			2  Rotterdam          9     2.5          7

			print "Inner join with merge()\n", pd.merge(dest, tips, how='inner')

			Out:
				Inner join with merge()
				        Dest  EmpNr  Amount
				0      Hague      5    10.0
				1  Rotterdam      9     5.0

			print "Outer join\n", pd.merge(dest, tips, how='outer')

			Out:
				Outer join
				        Dest  EmpNr  Amount
				0      Hague      5    10.0
				1  Amsterdam      3     NaN
				2  Rotterdam      9     5.0
				3        NaN      7     2.5

###4-9 处理数据缺失问题
* A:介绍
	* 数据记录中经常遇到空字段

* B:处理数据缺失
	* 考虑数据：
	* 
			import pandas as pd
			import numpy as np
			
			df = pd.DataFrame({'Country' : ['Afghanistan', 'Albania'],
			                    'Net primary enrolment ratia male (%)' :
			                     [np.NaN, 94]})
			print df

			Out:
				       Country  Net primary enrolment ratia male (%)
				0  Afghanistan                                   NaN
				1      Albania                                  94.0

	* pandas.isnull(dataframe) - 返回dataframe的isnull视图
	* 利用sum()进行统计
	* 
			print "NULL Values\n", pd.isnull(df)
			print "Total Null Values\n", pd.isnull(df).sum()

			Out:
				   Country  Net primary enrolment ratia male (%)
				0    False                                  True
				1    False                                 False
				Total Null Values
				Country                                 0
				Net primary enrolment ratia male (%)    1
				dtype: int64

	* pandas.notnull(dataframe) - 返回dataframe的isnull视图
	* 关于NaN值的运算，对乘法加法保持
	* dataframe.fillna(a) - 用一个变量值a替换NaN值
		* 
				print "Zero filled\n", df.fillna(0)
	
				Out:
					Zero filled
					       Country  Net primary enrolment ratia male (%)
					0  Afghanistan                                   0.0
					1      Albania                                  94.0

###4-10 处理日期数据
* A:日期处理问题
	* 日期数据处理比较复杂，如Y2K问题，悬而未决的2038年问题以及时区问题
	* pandas对此的作用
		* 帮助确定日期区间
		* 对时间序列数据重新采样
		* 对日期进行算数运算

* B:辅助确定日期区间
	* 判断日期是否超出范围
		* pandas.date_range('time', periods, freq) - 设定日期时间范围，freq为短码，返回日期List
		* 
				import pandas as pd
				import sys

				try:
				    print "Data range", pd.date_range('1/1/1677', period=4, freq='D')

				#学习：异常信息记录方式
				except:
				    etype, value, _ = sys.exc_info()
				    print "Error encountered", etype, value

				Out:
					Data range Error encountered <type 'exceptions.ValueError'> Must specify two of start, end, or periods

	* 计算允许的日期范围
	* 必须知道的条件
		* 1 日期的数据类型
			* pandas时间戳基于NumPy的datatime64类型，以纳秒为单位，用一个64位整数来表示具体数值
		* 2 精确时间中点
			* pandas的精确时间中点为1970年1月1日

		* pandas.DateOffset(seconds=) - 根据秒数创建pandas日期集合
		* pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, box=True, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix') - 将format符合规则的字符串转换成日期数据
			* arg可以为：
				* integer, float, string, datetime, list, tuple, 1-d array, Series 
			* 输出：
				* list-like: DatetimeIndex
				* Series: Series of datetime64 dtype
				* scalar: Timestamp
		* 
				offset = pd.DateOffset(seconds=2 ** 63/10 **9)
				mid = pd.to_datetime('1/1/1970')
				print "Start valid range", mid - offset
				print "End valid range", mid + offset

* B:对时间序列数据重新采样
* C:对日期进行算数运算

* D:附-pandas频率短码对照表：
	* 
			B business day frequency
			C custom business day frequency (experimental)
			D calendar day frequency
			W weekly frequency
			M month end frequency
			SM semi-month end frequency (15th and end of month)
			BM business month end frequency
			CBM custom business month end frequency
			MS month start frequency
			SMS semi-month start frequency (1st and 15th)
			BMS business month start frequency
			CBMS custom business month start frequency
			Q quarter end frequency
			BQ business quarter endfrequency
			QS quarter start frequency
			BQS business quarter start frequency
			A year end frequency
			BA business year end frequency
			AS year start frequency
			BAS business year start frequency
			BH business hour frequency
			H hourly frequency
			T, min minutely frequency
			S secondly frequency
			L, ms milliseconds
			U, us microseconds
			N nanoseconds

###4-11 数据透视表
* A:数据透视表的作用
	* 数据透视表可以用来汇总数据
	* 数据透视表可以从一个平面文件（现阶段所学的CSV文件）中指定的行和列中聚合数据（求和、平均值、标准差等）

* B:创建数据透视表
	* pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All') - 根据data来创建一个数据透视表，默认聚合函数为mean
	* 
			import pandas as pd
			from numpy.random import seed
			from numpy.random import rand
			from numpy.random import random_integers
			import numpy as np
			                                    
			seed(42)
			                                    
			df = pd.DataFrame({'Weather' : ['cold', 'hot', 'cold', 'hot', 'cold',
			            'hot', 'cold'],
			'Food' : ['soup', 'soup', 'icecream', 'chocolate',
			          'icecream', 'icecream', 'soup'],
			'price' : 10 * rand(7),
			'Number' : random_integers(1, 9, size=(7,))})
			print "DataFrame\n", df

			print pd.pivot_table(df, columns=['Food'], aggfunc=np.sum)

			Out:
								DataFrame
				        Food  Number Weather     price
				0       soup       8    cold  3.745401
				1       soup       5     hot  9.507143
				2   icecream       4    cold  7.319939
				3  chocolate       8     hot  5.986585
				4   icecream       8    cold  1.560186
				5   icecream       3     hot  1.559945
				6       soup       6    cold  0.580836
				Food    chocolate   icecream      soup
				Number   8.000000  15.000000  19.00000
				price    5.986585  10.440071  13.83338

###4-12 访问远程数据
* 不掌握

###总结
* 1.pandas的模块介绍和子库
* 2.pandas数据结构DataFrame，Sreies（介绍、创建方式、属性、共同功能）
* 3.pandas查询数据
* 4.DataFrame进行统计计算、数据聚合、数据透视表，处理缺失数据、处理日期数据
* 5.DataFrame的串联附加与连接