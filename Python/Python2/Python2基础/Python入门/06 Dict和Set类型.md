###6-1 Python之什么是dict
* A:什么是dict
 	我们已经知道，list 和 tuple 可以用来表示顺序集合，例如，班里同学的名字：
	* 
			['Adam', 'Lisa', 'Bart']

 	或者考试的成绩列表：
	* 
			[95, 85, 59]

 	但是，要根据名字找到对应的成绩，用两个 list 表示就不方便。
 	如果把名字和分数关联起来，组成类似的查找表：
	* 
			'Adam' ==> 95
			'Lisa' ==> 85
			'Bart' ==> 59

 	给定一个名字，就可以直接查到分数。
 	* Python的 dict 就是专门干这件事的。用 dict 表示“名字”-“成绩”的查找表如下：
 	* 
			d = {
			    'Adam': 95,
			    'Lisa': 85,
			    'Bart': 59
			}

	* 我们把名字称为key，对应的成绩称为value，dict就是通过 key 来查找 value。
	* 花括号 {} 表示这是一个dict，然后按照 key: value, 写出来即可。最后一个 key: value 的逗号可以省略。

* B:len()函数
	* 由于dict也是集合，len() 函数可以计算任意集合的大小：
	* 
			>>> len(d)
			3

	* 注意: 一个 key-value 算一个，因此，dict大小为3。

* C:任务
	* 新来的Paul同学成绩是 75 分，请编写一个dict，把Paul同学的成绩也加进去。
	* 
			d = {
			    'Adam': 95,
			    'Lisa': 85,
			    'Bart': 59
			}

###6-2 Python之访问dict
* A:案例演示
	* 我们已经能创建一个dict，用于表示名字和成绩的对应关系：
	* 
			d = {
			    'Adam': 95,
			    'Lisa': 85,
			    'Bart': 59
			}

 	那么，如何根据名字来查找对应的成绩呢？

* B:key的访问方式
	* 可以简单地使用 d[key] 的形式来查找对应的 value，这和 list 很像，不同之处是，list 必须使用索引返回对应的元素，而dict使用key：
	* 
			>>> print d['Adam']
			95
			>>> print d['Paul']
			Traceback (most recent call last):
			  File "index.py", line 11, in <module>
			    print d['Paul']
			KeyError: 'Paul'

	* 注意: 通过 key 访问 dict 的value，只要 key 存在，dict就返回对应的value。如果key不存在，会直接报错：KeyError。

	* 两个改进办法：
		* 一是先判断一下 key 是否存在，用 in 操作符：
		* 
				if 'Paul' in d:
				    print d['Paul']

 		如果 'Paul' 不存在，if语句判断为False，自然不会执行 print d['Paul'] ，从而避免了错误。

		* 二是使用dict本身提供的一个 get 方法，在Key不存在的时候，返回None：
		* 
				>>> print d.get('Bart')
				59
				>>> print d.get('Paul')
				None

* C:任务
	* 根据如下dict：
	* 
			d = {
			    'Adam': 95,
			    'Lisa': 85,
			    'Bart': 59
			}

	* 请打印出：
	* 
			Adam: 95
			Lisa: 85
			Bart: 59

###6-3 Python中dict的特点
* A:优缺点
	* 1.查找速度快
		* dict的第一个特点是查找速度快，无论dict有10个元素还是10万个元素，查找速度都一样。而list的查找速度随着元素增加而逐渐下降。
			* 不过dict的查找速度快不是没有代价的，dict的缺点是占用内存大，还会浪费很多内容，list正好相反，占用内存小，但是查找速度慢。
			* 由于dict是按 key 查找，所以，在一个dict中，key不能重复。

	* 2.key-value序对没有顺序
		* dict的第二个特点就是存储的key-value序对是没有顺序的！这和list不一样：
		* 
				d = {
				    'Adam': 95,
				    'Lisa': 85,
				    'Bart': 59
				}
		
		* 当我们试图打印这个dict时：
		* 
				>>> print d
				{'Lisa': 85, 'Adam': 95, 'Bart': 59}

 		打印的顺序不一定是我们创建时的顺序，而且，不同的机器打印的顺序都可能不同，这说明dict内部是无序的，不能用dict存储有序的集合。

	* 3.key 的元素必须不可变
 		* dict的第三个特点是作为 key 的元素必须不可变，Python的基本类型如字符串、整数、浮点数都是不可变的，都可以作为 key。但是list是可变的，就不能作为 key。
 		可以试试用list作为key时会报什么样的错误。
		* 不可变这个限制仅作用于key，value是否可变无所谓：
		* 
				{
				    '123': [1, 2, 3],  # key 是 str，value是list
				    123: '123',  # key 是 int，value 是 str
				    ('a', 'b'): True  # key 是 tuple，并且tuple的每个元素都是不可变对象，value是 boolean
				}

		* 最常用的key还是字符串，因为用起来最方便。

* B:任务
	* 请设计一个dict，可以根据分数来查找名字，已知成绩如下：
	* 
			Adam: 95,
			Lisa: 85,
			Bart: 59.

###6-4 Python更新dict
* A:添加键值对
	* dict是可变的，也就是说，我们可以随时往dict中添加新的 key-value。比如已有dict：
	* 
			d = {
			    'Adam': 95,
			    'Lisa': 85,
			    'Bart': 59
			}

	* 要把新同学'Paul'的成绩 72 加进去，用赋值语句：
	* 
			>>> d['Paul'] = 72

	* 再看看dict的内容：
	* 
			>>> print d
			{'Lisa': 85, 'Paul': 72, 'Adam': 95, 'Bart': 59}

* B:更新键值对
	* 如果 key 已经存在，则赋值会用新的 value 替换掉原来的 value：
	* 
			>>> d['Bart'] = 60
			>>> print d
			{'Lisa': 85, 'Paul': 72, 'Adam': 95, 'Bart': 60}

* C:任务
	* 请根据Paul的成绩 72 更新下面的dict：
	* 
			d = {
			    95: 'Adam',
			    85: 'Lisa',
			    59: 'Bart'
			}

###6-5 Python之遍历dict
* A:for循环遍历dict
	* 由于dict也是一个集合，所以，遍历dict和遍历list类似，都可以通过 for 循环实现。
	* 直接使用for循环可以遍历 dict 的 key：
	* 
			>>> d = { 'Adam': 95, 'Lisa': 85, 'Bart': 59 }
			>>> for key in d:
			...     print key
			... 
			Lisa
			Adam
			Bart

	* 由于通过 key 可以获取对应的 value，因此，在循环体内，可以获取到value的值。

* B:任务
	* 请用 for 循环遍历如下的dict，打印出 name: score 来。
	* 
			d = {
			    'Adam': 95,
			    'Lisa': 85,
			    'Bart': 59
			}

###6-6 Python中什么是set
* A:set特点
 	dict的作用是建立一组 key 和一组 value 的映射关系，dict的key是不能重复的。
 	有的时候，我们只想要 dict 的 key，不关心 key 对应的 value，目的就是保证这个集合的元素不会重复，这时，set就派上用场了。
	* set 持有一系列元素，这一点和 list 很像，但是set的元素没有重复，而且是无序的，这点和 dict 的 key很像。

* B:创建set
	* 创建 set 的方式是调用 set() 并传入一个 list，list的元素将作为set的元素：
	* 
			>>> s = set(['A', 'B', 'C'])

* C:查看set
	* 可以查看 set 的内容：
	* 
			>>> print s
			set(['A', 'C', 'B'])

	* 请注意，上述打印的形式类似 list， 但它不是 list，仔细看还可以发现，打印的顺序和原始 list 的顺序有可能是不同的，因为set内部存储的元素是无序的。

* D:自动去重复
	* 因为set不能包含重复的元素，所以，当我们传入包含重复元素的 list 会怎么样呢？
	* 
			>>> s = set(['A', 'B', 'C', 'C'])
			>>> print s
			set(['A', 'C', 'B'])
			>>> len(s)
			3

 	结果显示，set会自动去掉重复的元素，原来的list有4个元素，但set只有3个元素。

* E:任务
	* 请用set表示班里的4位同学：
	* 
			Adam, Lisa, Bart, Paul

###6-7 Python之访问set
* A:判断元素存在性
	* 由于set存储的是无序集合，所以我们没法通过索引来访问。
	* 访问set中的某个元素实际上就是判断一个元素是否在set中。
	* 例如，存储了班里同学名字的set：
	* 
			>>> s = set(['Adam', 'Lisa', 'Bart', 'Paul'])

	* 我们可以用 in 操作符判断：
		* Bart是该班的同学吗？
		* 
				>>> 'Bart' in s
				True

		* Bill是该班的同学吗？
		* 
				>>> 'Bill' in s
				False

		* bart是该班的同学吗？
		* 
				>>> 'bart' in s
				False

	* 看来大小写很重要，'Bart' 和 'bart'被认为是两个不同的元素。

* B:任务
	* 由于上述set不能识别小写的名字，请改进set，使得 'adam' 和 'bart'都能返回True。

###6-8 Python之set的特点
* A:set的特点
	* set的内部结构和dict很像，唯一区别是不存储value，因此，判断一个元素是否在set中速度很快。
	* set存储的元素和dict的key类似，必须是不变对象，因此，任何可变对象是不能放入set中的。
	* 最后，set存储的元素也是没有顺序的。

* B:set特点的应用
 	set的这些特点，可以应用在哪些地方呢？
 	* 案例演示
 		* 星期一到星期日可以用字符串'MON', 'TUE', ... 'SUN'表示。
		* 假设我们让用户输入星期一至星期日的某天，如何判断用户的输入是否是一个有效的星期呢？
			* 可以用 if 语句判断，但这样做非常繁琐：
			* 
					x = '???' # 用户输入的字符串
					if x!= 'MON' and x!= 'TUE' and x!= 'WED' ... and x!= 'SUN':
					    print 'input error'
					else:
					    print 'input ok'

				* 注意：if 语句中的...表示没有列出的其它星期名称，测试时，请输入完整。

			* 如果事先创建好一个set，包含'MON' ~ 'SUN'：
			* 
					weekdays = set(['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])

			* 再判断输入是否有效，只需要判断该字符串是否在set中：
			* 
					x = '???' # 用户输入的字符串
					if x in weekdays:
					    print 'input ok'
					else:
					    print 'input error'

 				这样一来，代码就简单多了。

* C:任务
	* 月份也可以用set表示，请设计一个set并判断用户输入的月份是否有效。
	* 月份可以用字符串'Jan', 'Feb', ...表示。

###6-9 Python之遍历set
* A:for循环遍历语法
	* 由于 set 也是一个集合，所以，遍历 set 和遍历 list 类似，都可以通过 for 循环实现。
	* 直接使用 for 循环可以遍历 set 的元素：
	* 
			>>> s = set(['Adam', 'Lisa', 'Bart'])
			>>> for name in s:
			...     print name
			... 
			Lisa
			Adam
			Bart

	* 注意: 观察 for 循环在遍历set时，元素的顺序和list的顺序很可能是不同的，而且不同的机器上运行的结果也可能不同。


* C:任务
	* 请用 for 循环遍历如下的set，打印出 name: score 来。
	* 
			s = set([('Adam', 95), ('Lisa', 85), ('Bart', 59)])

###6-10 Python之更新set
* A:更新原理
	* 由于set存储的是一组不重复的无序元素，因此，更新set主要做两件事：
	* 一是把新的元素添加到set中，二是把已有元素从set中删除。

* B:添加元素
	* 添加元素时，用set的add()方法：
	* 
			>>> s = set([1, 2, 3])
			>>> s.add(4)
			>>> print s
			set([1, 2, 3, 4])

	* 如果添加的元素已经存在于set中，add()不会报错，但是不会加进去了：
	* 
			>>> s = set([1, 2, 3])
			>>> s.add(3)
			>>> print s
			set([1, 2, 3])

* C:删除元素
	* 删除set中的元素时，用set的remove()方法：
	* 
			>>> s = set([1, 2, 3, 4])
			>>> s.remove(4)
			>>> print s
			set([1, 2, 3])

	* 如果删除的元素不存在set中，remove()会报错：
	* 
			>>> s = set([1, 2, 3])
			>>> s.remove(4)
			Traceback (most recent call last):
			  File "<stdin>", line 1, in <module>
			KeyError: 4

	* 所以用add()可以直接添加，而remove()前需要判断。

* D:任务
	* 针对下面的set，给定一个list，对list中的每一个元素，如果在set中，就将其删除，如果不在set中，就添加进去:
	* 
			s = set(['Adam', 'Lisa', 'Paul'])
			L = ['Adam', 'Lisa', 'Bart', 'Paul']

###总结
* 1.dict类型（概念，特点，访问，更新，遍历）
* 2.set类型（概念，特点，访问，更新，遍历）