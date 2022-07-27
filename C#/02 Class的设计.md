## 目录
* 1 构建类
* 2 数据成员与属性
* 3 索引器
* 4 成员初始化
* 5 `this`引用
* 6 静态成员
* 7 `const`和`readonly`成员
* 8 枚举类型
* 9 delegate类型
* 10 函数参数语义学
* 11 函数重载
* 12 变长参数列
* 13 操作符重载
* 14 转换式操作符

## 1 构建类
### 1.1 概念
**类（class）**是客观世界的抽象，其具体对象称为**实例（instance）**。

* 类的主要存在是状态和行为，对应面向对象设计中的**数据成员**和**方法成员**

	* 数据成员（作用域为类）又称为**字段**，不对外；方法成员又称为**成员函数**，可对外

	* **属性**是特殊的成员函数，用于安全修改类中的字段

	* 因此，对于外部观察者来说，类包含两个概念：属性和方法

下面以构建`WordCount`类为例，说明类型构建语法。

* 要点：

	* 确定数据成员和方法成员
	
	* 确定公开接口和细节隐藏

### 1.2 语法
(1) 确定操作集

这些操作将成为class的**成员函数**。

* `openFiles()` - 确认用户提供的文本文件的有效性，如果有效，就开启该文件。此外还开启一个输出文件用于保存单词计数。

* `readFile()` - 读取文本并存储，以备下一步处理。

* `countWords()` - 把文本分解成一个个单词，并统计单词出现的次数。

* `writeWords` - 按字典顺序将单词出现的次数输出到指定文件中。

除了这些操作外，还有一个**初始化任务**和**清理作业**，后面介绍。

(2) 确定成员函数

A 接口

对于每个成员函数而言，其接口（interface）由两部分组成：

* 函数返回值类型（return type）

* 函数参数列表（signature）

设计成员函数时，这些往往可以先省略，有助于简化编程。

B 级别

可以设置以下级别：

* `private` - 只能由class的其他成员函数调用

* `public` - 整个程序都可以访问

* `protected` - 同一个继承链上的成员函数可访问

考虑到面向对象的封装特性，结合各个成员函数的相互依存关系，

将整个调用序列封装为单个`public`函数`processFile()`，由它调用上述4个设置为`private`的成员函数。

### 1.3 例子
下面是WordCount Class的代码。

WordCount类：

```c#
using System;
public class WordCount
{
	public void processFile()
	{
		openFiles();
		readFile();
		countWords();
		writeWords();
	}

	private void countWords()
	{
		Console.WriteLine("！！！ WordCount.countwords()");
	}
	
	private void readFile()
	{
		Console.WriteLine("！！！ WordCount.readFile()");
	}
	
	private void openFiles()
	{
		Console.WriteLine("！！！ WordCount.openFiles()");
	}
	
	private void writeWords()
	{
		Console.WriteLine("！！！ WordCount.writeWords()");
	}
}
```

程序入口：

```c#
using System;
public class WordCountEntry
{
	public static void Main()
	{
		Console.WriteLine("Beginning WordCount program ...");

		WordCount theObj = new WordCount();
		theObj.processFile();

		Console.WriteLine("Ending WordCount program ...");
	}
}
```

## 2 数据成员与属性
### 2.1 概念
数据成员所表示的是：某个class的实体（instance）的相关状态信息，一般分为两类：

* 用户建立class实体是所提供的数据集

* 供多个成员函数实用的对象集

	* 若只供单一成员函数使用，可声明为local object

	* 若还供其他成员函数使用，可声明为class object

一般将字段（即数据成员中的class object）**声明为`private`级别**。

基于字段，可对外提供数据（字段集的子集，抽象为一个数据）的访问和修改功能，这样的数据及其对外函数的组合，称为**属性（property）**。

### 2.2 语法
(1) 声明字段

```c#
private 数据类型 标识符;
```

(2) 建立属性

```c#
访问级别 数据类型 标识符
{
	get
	{
		... ...
		reuturn 对应数据类型的值;
	}

	set
	{
		... ...
		字段 = value;
	}
}
```

**语法点：**

* `get`和`set`访问器可以两者只取其一，从而建立只读/只写property

### 2.3 例子
建立控制输出方式的类属性。

```c#
public class WordCount
{
	// private data member declaration
	private string m_file_output;

	// associated public property
	public string OutFile
	{
		get { return m_file_output; } // Read access
		set
		{
			// write access
			if ( value.Length != 0 )
				m_file_output = value;
		}
	}
}
```

## 3 索引器
### 3.1 概念
索引器（indexers）用来为class object提供诸如array一样的索引功能。

### 3.2 语法
indexer同property一样提供了get/set访问器，不同的是，这里应以关键字`this`标明indexer。

```c#
访问级别 数据类型 this[数据类型 标识符, 数据类型 标识符, ...] 
{
	get
	{
		... ...
		reuturn 对应数据类型的值;
	}

	set
	{
		... ...
		字段 = value;
	}
}
```

### 3.3 例子
建立矩阵类（Matrix）的二维indexer。

```c#
public class Matrix
{
	// not show: constructions, methods ...

	public int rows{ get{ return m_row; }}
	public int cols{ get{ return m_col; }}

	public double this[int row, int col]
	{
		get
		{
			check_bounds(row, col);
			return m_mat[row, col];
		}

		set
		{
			check_bounds(row, col);
			m_mat[row, col] = value;
		}
	}

	private int m_row;
	private int m_col;

	private double [,] m_mat; // 通常支持indexer者，都内含一个operator[]的容器
	private void check_bounds( int r, int c ) { ... }
}
```

## 4 成员初始化
### 4.1 概念
class的所有数据成员都会自动初始化为其类型的缺省值。

缺省初始化动作（default initialization）在operator `new`被调用时自动执行。

C#默认对各个类型设置了缺省值，也可以人为另外设置缺省值。

### 4.2 语法
(1) 方法一：类字段初始化

```c#
class 类名
{
	private 数据类型 标识符 = 缺省值;
	private 数据类型 标识符 = new 数据类型();
	// ...
}
```

(2) 方法二：构造方法初始化

```c#
public 类名(数据类型 参数1, 数据类型 参数2, ... ,数据类型 参数n)
{
	// 在此对字段进行初始化
}
```

此时建立类实例提供的参数，要匹配构造函数的参数列。

若还要使用其他构造函数，可以进行**重载（overloaded）**，即在建立一个**参数列表不同**的构造函数；或者调用其他构造函数，会先执行指定的构造函数，在执行本构造函数。

重载：

```c#
public 类名( 数据类型 参数1, 数据类型 参数2, ... , 数据类型 参数m)
{
	// 在此对字段进行初始化
}
```

调用其他构造函数：

```c#
public 类名( 数据类型 参数1, 数据类型 参数2, ... , 数据类型 参数m) : this(参数1, 参数2, ... , 参数m, ... 参数n)
{
	// 其他操作语句
}
```

### 4.3 例子
Point3D class演示构造函数的惯用手法。

```c#
Point3D origin = new Point3D(); // Point3D(0,0,0)
Point3D x_offset = new Point3D(1.0); // Point3D(1.0, 0, 0)
Point3D translate = new Point3D(1.0, 1.0); // Point3D(1.0, 1.0, 0)
Point3D mumble = new Point3D(1.0, 1.0, 1.0);

class Point3D
{
	public Point3D( double v1, double v2, double v3 )
	{
		x = v1; y = v2; z = v3;
	}

	public Point3D( double v1, double v2 ) : this(v1, v2, 0.0) {}
	public Point3D( double v1 ) : this(v1, 0.0, 0.0) {}
	public Point3D() this(0.0, 0.0, 0.0) {}

	// ...
}
```

## 5 `this`引用
### 5.1 概念
在类的实例中，`this`指向对象本身。

C#的编译和执行过程中，会对代码进行进行围绕`this`的修改。

对于“instance数据成员”，会加以扩展

```c#
x = new An_Object( y.z );

变为

this.x = new An_Object( this.y.z );
```

接着，对于成员函数，要相应的引入`this`

```c#
private void function ( ... ) { ... }

变为

private void function ( 类 this, ... ) { ... }
```

然后，对于该函数的调用，会传入`this`

```c#
theObj.function( ... );

变为

function( theObj, ... );
```

**提示点：**

* 在编写程序时，我们也可以直接使用`this`来实现需求。

## 6 静态成员
### 6.1 概念
数据成员和方法成员可以分为静态（static）成员和非静态成员。

非静态成员就是如之前所定义，是每一个实例特有的成员，属于实例成员；

静态成员是所有实例共有的成员，属于类成员。

### 6.2 语法
静态成员声明为`static`或`const`。

(1) static数据成员

static数据成员声明方法为：

```c#
访问级别 static 数据类型 标识符;
```

其有两种访问方式：

* 在class成员函数中，其访问语法与instance成员的访问语法一致

* 在class本体之外，可以通过class名称来访问

**语法点**

* static成员也可以使用第4节中的两种初始化方式来指定缺省值

**注意点：**

* 不能通过class名称访问instance成员；也不能通过instance名称访问static成员

(2) static构造函数

**语法点：**

* 每个class**至多可以定义一个**static构造函数

* 缺省情况下static构造函数的访问级别为public，且**不能显示指定**

* static构造函数的参数列表**必须是空的**

* static构造函数仅在“此class的某个实例被创建”或“此class的某个static成员被取用”，才会**调用且仅调用一次**

### 6.3 例子
WordCount中的文本分割字符集的声明：

```c#
public class WordCount
{
	static public char[] ms_separators;
	// ...
}
```

class成员函数中访问：

```c#
private void countWords()
{
	// ...
	for( int ix = 0; ix < m_text.Count; ++ix )
	{
		str = (string)m_text[ix];
		m_sentences[ix] = str.Split( ms_separators );
	}
}
```

class外类名直接访问：

```c#
static void Main()
{
	char bang = '!';
	int ix = 0;

	for( ; ix < WordCount.ms_separators.Length; ++ix )
		if( WordCount.ms_separators[ix] == bang )
			break;

	if( ix == WordCount.ms_separators.Length )
		throw new Exception("Insufficient separators");
}
```

## 7 `const`和`readonly`成员
### 7.1 `const`成员
#### 7.1.1 概念
使用`const`修饰的数据成员不可修改，否则会发生**编译期错误**。

#### 7.1.2 语法
const数据成员的声明必须包含初值，而且该初值必须是个常量表达式。

```c#
访问级别 const 数据类型 标识符 = value类数据;
```

**语法点：**

* const成员可以使用其他const成员初始化，前提是两者之间**没有循环依赖性**

* const成员本质上是**只读的staic成员**，具有static成员的访问语法

* const成员**不可能属于**reference类

* 唯一可以声明为`const`的reference类是**string**

#### 7.1.3 例子
合法的：

```c#
class Illustrate
{
	// OK: no circular dependency
	private const int x = y + z/2;
	private const int y = z * 2;
	private const int z = 4;
}
```

不合法的：

```c#
class Illustrate
{
	// error: cicular dependency
	private const int x = y + z/2;
	private const int y = z * 2;
	private const int z = x;
}
```

唯一一个可声明为`const`的refenence类：

```c#
class Illustrate
{
	// OK: string reference type is an exception
	private const string default_login = "Guest";
	private const string default_pswrd = "ChangeMe";
}
```

### 7.2 `readonly`成员
#### 7.2.1 概念
使用`readonly`修饰的数据成员不可修改，否则会发生**运行期错误**。

它与const成员的区别在于：

* readonly成员把object的初始化动作推迟到运行期进行

* readonly成员**不具有static性质**，但可以声明为static

#### 7.2.2 语法
readonly数据成员的声明与初始化可以用于reference类。

```c#
	访问级别 readonly 数据类型 标识符 = value类数据;
	访问级别 readonly 数据类型 标识符 = new 数据类型();
```

#### 7.2.3 例子
只有`readonly`才可用于refenence类：

```c#
public static readonly Matrix identity = new Matrix( 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 );
```

## 8 枚举类型
### 8.1 概念
枚举（enum）类型，定义了一组具有相互关联的具名整数常量，该类型的变量只能赋值为其中的具名常量之一。

使用enum类型的优点：

* 便于理解

* 限制数值范围

### 8.2 语法
定义

```c#
访问级别 enum 标识符 : 数据类型 {
	a = 1, b, c, d, e = a
}
```

访问

	标识符.枚举元;
	
	++标识符;
	
	标识符--;

**语法点：**

* 枚举元会自动递增1补全关联数值，默认为从0开始

* 缺省（`: 数据类型`）的情况下，使用`int`数据类型

### 8.3 例子
使用枚举类型完成星期的对接，并遍历枚举元。

```c#
public enum weekdays : byte
{
	sunday, monday, tuesday, wednesday,
	thursday, friday, staturday
};

public static void translator( string[] foregin )
{
	weekdays wd = weekdays.sunday;

	for ( ; wd <= weekdays.saturday; ++wd )
	{
		Console.WriteLine( wd + " : " + foreign[(int)wd] );
	}
}
```

## 9 delegate类型
### 9.1 概念
delegate类型用来指向一个或多个“具有特定标记式（signature）和返回类型的函数。

delegate类型主要有三个特征：

* delegate object可以同时代表（指向）多个成员函数，并可按照赋值的先后顺序调用

* delegate object代表的所有函数必须具有相同的原型（prototype）和标记（signature）【注：原型即返回值类型，标记即参数列表】

* 声明一个delegate类型就等于在内部创建了“.NET library framework”的`Delegate`或`MulticastDelegate`抽象基类的一个新子类实例、

### 9.2 语法
delegate类型的声明有四部分组成：

```c#
访问级别 delegate 返回类型 标识符();
```

调用delegate所指向的函数组：

```c#
标识符();
```

**语法点：**

* 如果某个 delegate 类型被用来同时代表两个或多个函数，其**返回类型必须是`void`**

### 9.3 例子
声明一个 delegate 类型：

```c#
public delegate void Action(); // 一个 delegate type
```

使用 delegate 声明的类型：

```c#
public class testHarness
{
	static private Action theAction; // 声明一个 delegate object
	static public Action Tester // 声明一个 delegate object的 property
	{
		get{ return theAction; }
		set{ theAction = value; }
	}
}
```

初始化 delegate object：

```c#
// in the testHarness Class
Annouce an = new Announce();
theAction = new Action( an.announceTime );

// suppleme: Announce Class
public class Announce
{
	public static void announceDate()
	{
		DateTime dt = DateTime.Now;
		Console.WriteLine( "Today's date is {0}", dt.ToLongDateString() );
	}

	publice void announceTime()
	{
		DateTime dt = DateTime.Now;
		Console.WriteLine( "The current time is {0}", dt.ToShortTimeString() );
	}
}
```

调用 delegate object：

```c#
testHarness.Tester();
```

异常处理：

```c#
// 在 calss之外
if( testHarness.Tester != null )
	testHarness.Tester();

// 在 class之内
static public void run()
{
	if( theAction != null)
		theAction();
}
```

添加/删除 delegate function：

```c#
public class testHashtable
{
	public void test0();
	public void test1();
	static testHashtable()
	{
		testHarness.Tester += new Action( test0 );
		testHarness.Tester += new Action( test1 );
		testHarness.Tester -= test1;
	}
}
```

查询 delegate object所代表的函数个数：

```c#
if ( testHarness.Tester != null && testHarness.Tester.GetInvocationList().Length != 0 )
{
	Action oldAct = testHarness.Tester;

	testHarness.Tester = act;
	testHarness.run();
	testHarness.Tester = oldAct;
}
else
{ ... }
```

## 10 函数参数语义学
### 10.1 概念
形参，函数定义中参数列表中的参数，是该函数的local object。

实参，调用函数是传入的参数。

每当外界调用函数，其形参就被绑定到实参上。

默认情况下，这一绑定动作以pass by value方式实现；使用`ref`或`out`修饰参数可改变该方式。

### 10.2 传值（Pass by Value）
#### 10.2.1 概念
缺省情况下，形参通过所谓的pass by value机制完成初始化，即每个形参将成为对应**实参值的一份副本**。

#### 10.2.2 语法
**语法点：**

* 如果形参是一种reference类别，pass-by-value机制有点特别

	* 首先一个独立的local实体会被创建出来，复制reference handle的值

	* 不同的是，形参和实参都指向heap上的同一个object

	* 这意味着对heap object所做的改动是永久的；对handle object的改动是独立的

#### 10.2.3 例子
下面举一个例子说明 reference 类别的传值机制：

```c#
static public void byValue( string s )
{
	Console.Write("\nInside byValue: ");
	Console.WriteLine("original parameter: \n\t" + s);

	// now refers to a different string object!
	s = s.ToUpper();

	Console.Write("\nInside byValue: ");
	Console.WriteLine("modified parameter: \n\t" + s);
}

string s = "A fine and private place";
Console.WriteLine("string to be passed by value: \n\t" + s);
byValue(s);
Console.WriteLine("back from call -- string: \n\t" + s);
```

从输出就可以看出来端倪：

```c#
string to be passed by value: 
	A fine and private place

Inside byValue: original parameter:
	A fine and private place

Inside byValue: modified parameter:
	A FINE AND PRICATE PLACE

back from call -- string:
	A fine and private place
```

### 10.3 传址（Pass by Reference）：ref参数
#### 10.3.1 概念
ref参数类似于将形参变为实参的别名（alias），函数中对reference参数的修改，将直接改动相应的实参。

#### 10.3.2 语法
在相应的形参和实参前添加`ref`修饰即可。

```c#
函数名(ref 数据类型 标识符) // 形参修饰语法

函数名( ref 标识符 ); // 实参修饰语法
```

**注意点：**

* 该方式**不会触发**自动装箱

#### 10.3.3 例子
沿用上面的例子，使用 ref 参数后：

```c#
string to be passed by value: 
	A fine and private place

Inside byValue: original parameter:
	A fine and private place

Inside byValue: modified parameter:
	A FINE AND PRICATE PLACE

back from call -- string:
	A FINE AND PRICATE PLACE
```

### 10.4 传址（Pass by Reference）：out参数
#### 10.4.1 概念
out参数允许在函数体内对其进行赋值，但**不能使用传入的值**。主要用于实现函数返回多个参数。

ref参数与out参数的区别：**ref参数有进有出，out参数只出不进**。

#### 10.4.2 语法
其语法与上一个类似。

```c#
out 数据类型 标识符 // 形参修饰
函数( out 标识符 ); // 实参修饰
```

**注意点：**

* 编译器要求每个out参数在函数内的每个退出点都要被赋值

* ref与out参与函数重载决议（关于重载，见11）

## 11 函数重载
### 11.1 概念
两个或多个名称可以共用同一个名称，而用函数参数列在**参数类型和参数个数**加以区分，这样的行为称为函数重载。

### 11.2 语法
* 确认函数调用动作所对应的“重载函数集”（overload functions set），其中所有函数称为“**候选**函数”

* 根据参数列的参数类型和参数个数是否合适，选出“**可行**函数”

	* 若实参类型与形参类型不匹配，就要求实参到形参**存在隐式类型转换**

	* 若有`ref`和`out`参数，对应的实参类型和形参类型**必须完全匹配**（因为不进行自动类型转换/自动装箱拆箱）

* 从可行函数中选出**最佳**匹配

	* 如果T1至T2的某个隐式转换存在，而且有T2至T1的隐式转换不存在，那么“S转为T1”就是“较佳的”

	* 若仍然无法比较（存在模棱两可的情况），会导致编译错误，可对实参进行显式转换

### 11.3 例子
以下模棱两可的重载会导致错误：

```c#
public static void g( long l, float f ) { ... }
public static void g( int i, double d ) { ... }

g(0, 0); // (int, int)，ambiguous
```

对第一个参数来说，函数2完全匹配，函数1需要隐式转换；对第二个参数来说，函数1隐式转换更好。两者造成歧义。

可以做显示转换：

```c#
g( 0, (double)0 );
```

## 12 变长参数列
### 12.1 概念
变长参数允许传入任意数目的参数，甚至类型也可以是任意的。

### 12.2 语法
定义变长参数：

```c#
函数名部分( 其他参数 ..., params 指定类型[] 参数名 ) { ... }
```

**注意点：**

* 一个函数只能带有一个params array，而且必须在参数列**最末尾处声明**

* params array**只能是一维**，而且不能以关键字`ref`或`out`修饰

**提示点：**

* 将指定类型声明为`Object`可以接受任意类型的参数

* 可以使用`foreach`语句遍历参数，也可以根据数组长度用`for`语句遍历

### 12.3 例子
一个例子：

```c#
public void message( string msg, params object[] args )
{
	Console.WriteLine( msg );

	if( args.Length != 0 )
		foreach( object o in args )
			Console.WriteLine( "\t{0}", o.ToString() );
}
```

## 13 操作符重载
### 13.1 概念
操作符的行为也能够重载，而不必提供具名函数。

### 13.2 语法
一种可能的重载做法：

```c#
public static 类名 operator操作符 (操作符要求参数) { ... }
```

以下是8个可被重载的一元操作符：

```c#
+, -, !, ~, ++, --, true, flase
```

其中被重载了`true`或`false`操作符的类，其引用变量就可以直接作为判别条件。、

一下是16个可被重载的二元操作符：

```c#
+, -, *, /, %, &, |, ^, <, >, <<, >>, ==, !=, <=, >=
```


**语法点：**

* 必须声明为public 和static ，重载的运算符以operator关键字引出，参数至少包含一个本类的实例

* 被重载的运算符操作的参数不能是`ref`或`out`形式的

* 一旦提供了某个操作符的重载实体，则该操作符对应的“复式赋值操作符”（如果有）的支持会自动实现，并且不允许重载复式赋值操作符

* `true`和`false`、`<<`和`>>`、`==`和`!=`、`<=`和`>=`必须成对重载

* `++`操作符和`--`操作符必须返回其所隶属的class的一个object

**注意点：**

* 被重载的操作符的参数不可以是`ref`或`out`

### 13.3 例子
实现矩阵（Matrix）乘法的一个可能例子：

```c#
public class Matrix
{
	public static Matrix operator* (Matrix mat, double dval)
	{
		Matrix result = new Matrix( mat.rows, mat.cols );

		for( int ix = 0; ix < mat.rows; ix++ )
			for(int iy = 0; iy < mat.cols; iy++ )
				result[ix,iy] = mat[ix,iy] * dval;

		return result;
	}
	// ... rest of the Matrix class
}
```

## 14 转换式操作符
### 14.1 概念
C#提供该种机制，可以让每个class都定义一组“可应用于其object身上”的隐式（implicit）或显式（explicit）转换。

即可以自定义类的隐式转换和显式转换机制。

### 14.2 语法
隐式转换

```c#
public static implicit operator 返回类型 ( 原始类型 ) { ... }
```

显式转换

```c#
public static explicit operator 返回类型 ( 原始类型 ) { ... }
```

**语法点：**

* 不能将两个两种方向的转换都定义为`implicit`，这是可指定其中一个为`explicit`

**注意点：**

* 参数既不能是`ref`参数也不能是`out`参数

**提示点：**

* 经验显式，其他类型“转换进来”比本类型“转换出去”，**不成功**的可能性会更大些，故“转换进来”用显式转换，“转换出去”用隐式转换

### 14.3 例子
BitVector是一个bit序列的向量类，使其支持隐式转换操作：

```c#
public class BitVector
{
	static public implicit operator string( BitVector bv ){ ... }

	static public implicit operator ulong( BitVector bv ){ ... }

	// ... rest of BitVector class definition
}
```

支持显示转换：

```c#
static public explicit operator BitVector( string s ){ ... }

static public explicit operator BitVector( ulong ul ){ ... }
```