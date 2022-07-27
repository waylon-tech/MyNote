###1-1 本书用到的软件
* A:数据分析软件脑图
 	![图片被吞掉了]()

* B:程序库介绍
	* NumPy - 一个基础性的Python库，为我们提供了常用的数值数组和函数
	* SciPy - Python的科学计算库，对NumPy的功能进行了补充，同时有部分功能重合
	* matplolib - 一个基于NumPy的绘图库
	* IPython 为交互式计算提供了一个基础设施

* C:Windows Linux MasOSX 平台安装库
	* 1.网站下载安装
	* 2.从源代码安装
	* 3.settools安装

###1-2 NumPy数组简介
* A:基于向量化的运算
	* 与Python中的列表相比，进行数据运算时，效率更高

* B:一个简单应用
	* 解决向量加法问题
		* 纯Python语言
		* 
				def pythonsum(n):
					a = range(n)
					b = range(n)
					c = []
				for i in range(len(a)):
					a[i] = i ** 2
					b[i] = i ***3
					c.append(a[i] + b[i])
				return c

		* 利用NumPy
		* 
				def numpysum(n):
					a = numpy.arange(n) ** 2
					b = numpy.arange(n) ** 3
					c = a + b
				return c

		* 注：
			* arange(n) - 创建一个含有整数0到n的NumPy数组

	* 运行
		* 
				import sys
				from datetime import datetime
				import numpy as np
				
				//两个算法
	
				size = int(sys.argv[1])
	
				start = datetime.now()
				c = pyhtonsum(size)
				delta = datetime.now - start
				print "The last two elements of the sum", c[-2:]
				print "Python elapsed time in microseconds", delta,microseconds
	
				start = datetime.now()
				c = numpysum(size)
				delta = datetime.now - start
				print "The last two elements of the sum", c[-2:]
				print "Python elapsed time in microseconds", delta,microseconds

	* NumPy的速度比等价的常规Python代码要快很多

###1-3 将IPython用作shell
* 暂时不用

###总结
* 1.介绍了软件、程序库和安装方法
* 2.以NumPy数组为例，对库的作用进行了介绍
* 3.IPython作为shell的用法