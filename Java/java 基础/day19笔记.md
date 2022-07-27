###19.01_异常(异常的概述和分类)
* A:异常的概述
	* 异常就是Java程序在运行过程中出现的错误。
* B:异常的分类
	* 通过API查看Throwable
	* Error
		* 服务器宕机,数据库崩溃等
	* Exception
C:异常的继承体系
	* Throwable
		* Error	
		* Exception
			* RuntimeException

###19.02_异常(JVM默认是如何处理异常的)
* A:JVM默认是如何处理异常的
	* main函数收到这个问题时,有两种处理方式:
	* a:自己将该问题处理,然后继续运行
	* b:自己没有针对的处理方式,只有交给调用main的jvm来处理
	* jvm有一个默认的异常处理机制,就将该异常进行处理.
	* 并将该异常的名称,异常的信息.异常出现的位置打印在了控制台上,同时将程序停止运行
* B:案例演示
	* JVM默认如何处理异常
	* 如何分析：从后往前看

###19.03_异常(try...catch的方式处理异常1)
* A:异常处理的两种方式
	* a:try…catch…finally 
		* try：用来检测异常 catch：用来捕获异常 finally：释放资源
		* try catch
		* try catch finally
		* try finally 
	* b:throws
* B:try...catch处理异常的基本格式
	* try…catch…finally
* C:案例演示
	* try...catch的方式处理1个异常
* 同一个包下不能有两个相同的源文件

###19.04_异常(try...catch的方式处理异常2)
* A:案例演示
	* try...catch的方式处理多个异常
	* 小的异常放前面，大的异常放后面
	* JDK7以后处理多个异常的方式及注意事项：
	* catch (A a | B b) {处理代码}

###19.05_异常(编译期异常和运行期异常的区别)
* A:编译期异常和运行期异常的区别
	* Java中的异常被分为两大类：编译时异常和运行时异常。
	* 所有的RuntimeException类及其子类的实例被称为运行时异常，其他的异常就是编译时异常
	
	* 编译时异常
		* Java程序必须显示处理，否则程序就会发生错误，无法通过编译
	* 运行时异常
		* 无需显示处理，也可以和编译时异常一样处理
* B:案例演示
	* 编译期异常和运行期异常的区别

###19.06_异常(Throwable的几个常见方法)
* A:Throwable的几个常见方法
	* a:getMessage()
		* 获取异常信息，返回字符串。
	* b:toString()
		* 获取异常类名和异常信息，返回字符串。
	* c:printStackTrace()
		* 获取异常类名和异常信息，以及异常出现在程序中的位置。返回值void。
* B:案例演示
	* Throwable的几个常见方法的基本使用

###19.07_异常(throws的方式处理异常)
* A:throws的方式处理异常
	* 定义功能方法时，需要把出现的问题暴露出来让调用者去处理。
	* 那么就通过throws在方法上标识。
	* 运行时异常RunTimeException，在方法上不用作声明
* B:案例演示
	* 举例分别演示编译时异常和运行时异常的抛出

###19.08_异常(throw的概述以及和throws的区别)
* A:throw的概述
	* 在功能方法内部出现某种情况，程序不能继续运行，需要进行跳转时，就用throw把异常对象抛出。
* B:案例演示	
	* 分别演示编译时异常对象和运行时异常对象的抛出
* C:throws和throw的区别 (位置，名字，个数)
	* a:throws
		* 用在方法声明后面，跟的是异常类名
		* 可以跟多个异常类名，用逗号隔开
		* 表示抛出异常，由该方法的调用者来处理
	* b:throw
		* 用在方法体内，跟的是异常对象名
		* 只能抛出一个异常对象名
		* 表示抛出异常，由方法体内的语句处理

###19.09_异常(finally关键字的特点及作用)
* A:finally的特点
	* 被finally控制的语句体一定会执行
	* 特殊情况：在执行到finally之前jvm退出了(比如System.exit(0))，这就不行了！
* B:finally的作用
	* 用于释放资源，在IO流操作和数据库操作中会见到
* C:案例演示
	* finally关键字的特点及作用

###19.10_异常(finally关键字的面试题)
* A:面试题1
	* final,finally和finalize的区别
	* fianl：可以修饰类，本能被继承；方法，不能被重写；变量，只能赋值一次
	* finally：是try语句中的一个语句体，不能单独使用，用来释放资源
	* finalize：是一个方法，当垃圾回收器确定不存在该对象的更多引用时，由对象的垃圾回收器调用此方法
* B:面试题2
	* 如果catch里面有return语句，请问finally的代码还会执行吗?如果会，请问是在return前还是return后。
	* return是已经进行变量装填了，return得不彻底
	* 不要在fianlly里写return，finally肯定会执行，它会改变try和catch的结果
	
###19.11_异常(自定义异常概述和基本使用)
* A:为什么需要自定义异常
	* 为了通过名字区分到底是什么异常，有针对的解决办法
	* 举例：人的年龄
* B:自定义异常概述
	* 继承自Exception
	* 继承自RuntimeException
* C:案例演示
	* 自定义异常的基本使用

###19.12_异常(异常的注意事项及如何使用异常处理)
* A:异常注意事项
	* a:子类重写父类方法时，子类的方法必须抛出相同的异常或父类异常的子类。(父亲坏了,儿子不能比父亲更坏)
	* b:如果父类抛出了多个异常,子类重写父类时,只能抛出相同的异常或者是他的子集,子类不能抛出父类没有的异常
	* c:如果被重写的方法没有异常抛出,那么子类的方法绝对不可以抛出异常,如果子类方法内有异常发生,那么子类只能try,不能throws
* B:如何使用异常处理
	* 原则:如果该功能内部可以将问题处理,用try,如果处理不了,交由调用者处理,这是用throws
	* 区别:
		* 后续程序需要继续运行就try
		* 后续程序不需要继续运行就throws
		
	* 如果JDK没有提供对应的异常，需要自定义异常。

###19.13_异常(练习)
* 键盘录入一个int类型的整数,对其求二进制表现形式
	 * 如果录入的整数过大,给予提示,录入的整数过大请重新录入一个整数BigInteger
	 * 如果录入的是小数,给予提示,录入的是小数,请重新录入一个整数
	 * 如果录入的是其他字符,给予提示,录入的是非法字符,请重新录入一个整数
* 快捷键：Alt + Shift + z 捕捉异常

###19.14_File类(File类的概述和构造方法)
* A:File类的概述
	* File更应该叫做一个路径
		* 文件路径或者文件夹路径  
		* 路径分为绝对路径和相对路径
		* 绝对路径是一个固定的路径,从盘符开始
		* 相对路径相对于某个位置,在eclipse下是指当前项目下,在dos下指的是当前路径
	* 查看API的Flie类
	* 文件和目录路径名的抽象表示形式
* B:构造方法
	* File(String pathname)：根据一个路径得到File对象 (\表示转义字符，\\表示反斜线)
	* File(String parent, String child):根据一个目录和一个子文件/目录得到File对象
	* File(File parent, String child):根据一个父File对象和一个子文件/目录得到File对象 (Flie封装比String更强大一点，可使用Flie里面的方法)
* C:案例演示
	* File类的构造方法

###19.15_File类(File类的创建功能)
* A:**创建功能**
	* public boolean createNewFile():创建文件 如果存在这样的文件，就不创建了
	* public boolean mkdir():创建文件夹 如果存在这样的文件夹，就不创建了
	* public boolean mkdirs():创建文件夹,如果父文件夹不存在，会帮你创建出来
* B:案例演示
	* File类的创建功能，后缀无关要紧，只是习惯而已

	* 注意事项：
		* 如果你创建文件或者文件夹忘了写盘符路径，那么，默认在项目路径下。
		
###19.16_File类(File类的重命名和删除功能)
* A:**重命名和删除功能**
	* public boolean renameTo(File dest):把文件重命名为指定的文件路径
	* public boolean delete():删除文件或者文件夹
* B:重命名注意事项
	* 如果路径名相同，就是改名。
	* 如果路径名不同，就是改名并剪切。
* C:删除注意事项：
	* Java中的删除不走回收站。
	* 要删除一个文件夹，请注意该文件夹内不能包含文件或者文件夹，否则无法删除


###19.17_File类(File类的判断功能)
* A:**判断功能**
	* setReadable():设置是否可读 但，windows系统认为所有文件都是可读的
	* setWritable():设置是否可写 windows系统可以设置为不可写
	* public boolean isDirectory():判断是否是目录（文件夹）
	* public boolean isFile():判断是否是文件
	* public boolean exists():判断是否存在
	* public boolean canRead():判断是否可读
	* public boolean canWrite():判断是否可写
	* public boolean isHidden():判断是否隐藏
* B:案例演示
	* File类的判断功能
	
###19.18_File类(File类的获取功能)
* A:**获取功能**
	* public String getAbsolutePath()：获取绝对路径
	* public String getPath():获取路径 获取构造方法里面传入的路径
	* public String getName():获取名称 获取文件或文件夹的名称
	* public long length():获取长度（字节数）
	* public long lastModified():获取最后一次的修改时间，毫秒值(相对于1970.01.01 00:00:00)
	* public String[] list():获取指定目录下的所有文件或者文件夹的名称数组(只是名字)
	* public File[] listFiles():获取指定目录下的所有文件或者文件夹的File数组(全路径)
* B:案例演示
	* File类的获取功能

###19.19_File类(输出指定目录下指定后缀的文件名)
* A:案例演示
	* 需求：判断E盘目录下是否有后缀名为.jpg的文件，如果有，就输出该文件名称
* 方法一：
	* list() String类的endsWith()方法
	*  listFiles() String类的endsWith()方法
* 方法二：过滤器↓

###19.20_File类(文件名称过滤器的概述及使用)
* A:文件名称过滤器的概述
	* public String[] list(FilenameFilter filter)
	* public File[] listFiles(FileFilter filter)
* B:文件名称过滤器的使用
	* 需求：判断E盘目录下是否有后缀名为.jpg的文件，如果有，就输出该文件名称
* C:源码分析
	* 带文件名称过滤器的list()方法的源码

###19.21_File类(递归)
* 5的阶乘
* 留到IO最后一天讲

###19.22_day19总结
把今天的知识点总结一遍。
一、异常（概述，分类，try..，throw..，finally，自定义及使用）
二、File类（概述、创建、重命名和删除、判断、获取功能，过滤代码实现）