###1-1 正则简介
* A:正则表达式基本概念
	* 为什么使用正则
		* 案例1:找到"imooc"开头的信息
			* 使用字符串匹配就可以实现
			* 
					def find_start_imooc(fname):

		* 案例2:找到以"imooc"开头和结尾的信息
			* 
					def 

		* 案例3:匹配一个下划线和字母开头的变量名
			* 
					def

		* 每次匹配都要单独完成，所以抽象数一个规则

	* 使用单个字符串来描述匹配一系列符合某个句法规则的字符串
	* 是对字符串操作的一种逻辑公式
	* 应用场景：处理文本和数据

###2-1 re模块的使用
* A:概要
	* 1.导入re模块
		* 
				import re

	* 2.正则表达式 --> compile --> pattern --> match --> 结果

* B:案例演示
	* 
			strl1 = 'imooc python'
			str1.find('11')		#返回-1
			str1.find('imooc')		#返回0
			str1.startswith('imooc')	#返回True
			
			import re
			pa = re.compile(r'imooc')	#返回pattern对象
	
			ma = pa.match(str1)		#返回一个match对象
	
			ma.group()	#返回'imooc'字符串或元组
			ma.span()	#返回(0,5) 位置
			ma.string()	#返回'imooc python'

	* 方法1：分步
		* pa = re.compile(r'...')	compile定义正则表达式，得到pattern对象
		* ma = pa.match('...')		match匹配字符串，得到match对象
		* ma.group()				返回匹配的字符串
		* ma.groups()				返回匹配的元组
		* ma.span()					返回匹配字符串或元组的位置
		* ma.string()				返回要匹配的字符串

	* 方法2：一步
		* re.match(pattern, string, flags=0)
		* 只适用于匹配一次的情况

* C:compile()详解
	* re.compile(r'...', re.I)	re.IGNORECASE - 忽略大小写

###1-3 正则表达式基本语法
* 正则语法与其他语言通用
* A:匹配单个字符
	* .				- 匹配任意字符（处除了\n）
	* [...]			- 匹配字符集中的任意一个字符
	* \d 与 \D		- 匹配数字集 与 匹配非数字集
	* \s 与 \S		- 匹配空白集 与 非空白字符集
	* \w 与 \W		- 匹配单词字符集[a-zA-Z0-9] 与 非单词字符集

	* 注意：转义，如\[ \]

* B:匹配多个字符
	* `*`			- 匹配前一个字符0次或者无数次
	* + 			- 匹配前一个字符1次或者无数次
	* ?				- 匹配前一个字符0次或者1次
	* {m} 与 {m,n}	- 匹配前一个字符m次或者m到n次
	* *? 与 +? 与 ??	- 匹配模式变为非贪婪（尽可能少匹配字符）

* C:边界匹配
	* ^				- 匹配字符串开头
	* $				- 匹配字符串结尾
	* \A 与 \Z		- 指定的字符串必须出现在开头或结尾

* D:分组匹配
	* |				- 匹配左右任意一个表达式
	* (ab)			- 括号中表达式作为一个分组
	* \<number>		- 引用编号为num的分组匹配到的字符串 自动编号？
	* (?P<name>)	- 给分组起个别名
	* (?P=name)		- 引用别名为name的分组匹配到的字符

* E:补充
	* 非贪婪模式：只匹配pattern能匹配到的最少字符，便不再往后匹配
	* 意义：如果一个表达式中有多个未知匹配次数的表达式，应防止进行不必要的尝试匹配。

###1-4 re模块相关方法
* A:re模块的方法
	* 1.search(pattern, string, flag=0)
		* 在一个字符串中查找匹配，返回第一次出现的索引位置
		* 相比：find()方法查找固定字符串
	
	* 2.findall(pattern, string. flag=0)
		* 找到匹配，返回所有匹配部分的列表

	* 3.sub(pattern, repl, string, count=0, flags=0)
		* 将字符串中匹配正则表达式的部分替换为其他值，返回字符串

	* 4.split(pattern, string, maxsplit=0, flags=0)
		* 根据匹配分割字符串，返回分割字符串组成的列表
		* 相比：字符串的spile()方法按空格分割

###1-5 练习：爬取网站图片到本地
* A:过程分析
	* 1.抓去网页
	* 2.获取图片地址
	* 3.抓去图片内容并保存到本地

* B:案例演示
* 
		import urllib2     							#导入urllib2包
		req=urllib2.urlopen('http://www.XXX.com')  #对网页发出请求
		buf=req.read()    							#将网页内容读取到buf缓存中
		import re
		urllist = re.findall(r'http:.+\.jpg',buf)   #获取图片地址列表
		i=0
		for url in urllist:
		    f=open(str(i)+'.jpg','w')    			#创建一个文件，命名为{str(i).jpg}    ‘w’:采用写入方式，若无该文件则创建它
		    req = urllib2.urlopen(url)     			#请求该地址内容
		    buf = req.read()     					#读取请求信息
		    f.write(buf)         					#将buf内容写到文件f中
		    i+=1

###总结
* 1.正则表达式概念
* 2.re模块的使用
* 3.正则表达式的基本语法
* 4.正则表达式相关方法
* 5.爬取网页练习