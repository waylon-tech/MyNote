# Matlab创建函数的三种方法

## （1）inline函数：
* 格式
* 
		f=inline('3*x+1','x')

* 可以得到：
* 	
	```matlab
	f=
	Inline function :
f(x)=3*x+1
	```
	
* 输入t=0:3;
* 
		f(t)
		
	```matlab
	ans =
1 4 7 10
		```
		
* 注意
	
	* 在未来的版本将被匿名函数替换

## （2）匿名函数：
* 格式
* 
		f=@(x)3*x+1

* 可以得到：
* f=
			@(x)3*x+1
		
	
	```matlab
	输入t=0:3;
	f(t)
	
	ans =
	1 4 7 10
	```

## （3）创建M-函数
* 格式
* 新建m文件，输入：
* 
		function f=equation(x)
		f=3*x+1;

* 保存m文件到工作文件夹；

* 输入t=0:3;
		f=equation(t)
		
	```matlab
	ans =
	1 4 7 10
	```