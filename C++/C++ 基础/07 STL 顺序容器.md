## 目录

[toc]

### 7.1 顺序容器

#### 7.1.1 顺序容器概述

| 顺序容器       | 说明                                                         | 扩充规则                   |
| -------------- | ------------------------------------------------------------ | -------------------------- |
| `vector`       | 可变大小数组。支持快速随机访问，在尾部之外的位置插入或删除元素可能很慢 | 每次扩充一倍               |
| `deque`        | 双端队列。支持快速随机访问，在头尾位置插入/删除速度很快      | 每次扩充一个 `buffer(512)` |
| `list`         | 双向链表。只支持双向顺序访问，在 `list` 中任何位置进行插入/删除操作速度都很快 | 每次扩充一个元素           |
| `forward_list` | 单向链表。只支持单向顺序访问，在链表任何位置进行插入/删除操作速度都很快 | 每次扩充一个元素           |
| `array`        | 固定大小数组。支持快速随机访问，不能添加和删除元素           | 固定大小                   |
| `string`       | 字符串。与 `vector` 相似的容器，但专门用于保存字符，随机访问快，在尾部插入/删除速度快 | 同 `vector`                |

【注1】每个容器都定义在一个头文件中，文件名与类型名相同。

【注2】新标准库容器的性能几乎肯定与最精心优化过的同类数据结构一样好。

【注3】容器的选择取决于占主导地位的操作，因此可以尝试对容器应用性能进行测试。

#### 7.1.2 顺序容器操作

一般容器的操作：`6.3.3 一般容器操作`；

##### 7.1.2.1 添加元素

<table>
    <tr>
    	<th>顺序容器添加操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c.push_back(t)</td>
        <td rowspan=2>
            在 c 的尾部创建一个值为 t 或由 args 创建的元素，返回 void；<br/>
            array, forward_list 不支持这两个操作；
        </td>
    </tr>
    <tr>
    	<td>c.emplace_back(args)</td>
    </tr>
    <tr>
    	<td>c.push_front(t)</td>
        <td rowspan=2>
            在 c 的头部创建一个值为 t 或由 args 创建的元素，返回 void；<br/>
            array 不支持这两个操作；vector 和 string 不支持这两个操作，尽管用 insert 可以实现这个效果，但是非常耗时；
        </td>
    </tr>
    <tr>
    	<td>c.emplace_front(args)</td>
    </tr>
    <tr>
    	<td>c.insert(p,t)</td>
        <td rowspan=2>
            在迭代器 p 指向的元素之前创建一个值为 t 或由 args 创建的元素，返回指向新添加的元素的迭代器；<br/>
            array 不支持这两个操作；forward_list 不支持这两个操作，它有自己专门版本的 insert 和 emplace；
        </td>
    </tr>
    <tr>
    	<td>c.emplace(p,args)</td>
    </tr>
    <tr>
        <td>c.insert(p,n,t)</td>
        <td>
            在迭代器 p 指向的元素之前插入 n 个值为 t 的元素，返回指向新添加的第一个元素的迭代器，若 n 为 0，则返回 p；<br/>
            array 不支持这个操作；forward_list 不支持这个操作，它有自己专门版本的 insert；
        </td>
    </tr>
    <tr>
    	<td>c.insert(p,b,e)</td>
        <td>
            将迭代器 b 和 e 指定的范围内的元素插入到迭代器 p 指向的元素之前（b 和 e 不能指向 c 中的元素），返回指向新添加的元素的第一个元素的迭代器，若范围空，则返回 p；<br/>
            array 不支持这个操作；forward_list 不支持这个操作，它有自己专门版本的 insert；
        </td>
    </tr>
    <tr>
    	<td>c.insert(p,il)</td>
        <td>
            il 是一个花括号包围的元素值列表，将这些给定值插入到迭代器 p 指向的元素之前，返回指向新添加的第一个元素的迭代器，若列表为空，则返回 p；<br/>
            array 不支持这个操作；
        </td>
    </tr>
</table>

【注1】添加元素操作会改变容器的大小，对于 `vector`、`string` 和 `deque` 来说，插入元素会使所有指向容器的迭代器、引用和指针失效。

【注2】`emplace_front`、`emplace` 和 `emplace_back` 中三个函数对应 `push_front`、`insert` 和 `push_back`，它们的区别是：

* 前者将参数 `args` 与元素类型的构造函数<u>匹配</u>，直接在容器中<u>构造</u>对象
* 后者将对象<u>拷贝</u>到一个局部临时对象，压入到容器中

##### 7.1.2.2 删除元素

<table>
    <tr>
    	<th>顺序容器删除操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c.pop_back()</td>
        <td>
            删除 c 中尾元素。若 c 为空，则函数行为未定义。函数返回 void；<br/>
            array, forward_list 不支持这个操作；
        </td>
    </tr>
    <tr>
    	<td>c.pop_front()</td>
        <td>
            删除 c 中首元素。若 c 为空，则函数行为未定义。函数返回 void；<br/>
            array 不支持这个操作；vector 和 string 不支持这个操作，尽管用 erase 可以实现这个效果，但是非常耗时；
        </td>
    </tr>
    <tr>
    	<td>c.erase(p)</td>
        <td>
            删除迭代器 p 所指向的元素，返回一个指向被删元素之后元素的迭代器；<br/>
        	若 p 指向尾元素，则函数返回尾后迭代器；若 p 是尾后迭代器，则函数行为未定义；<br/>
            array 不支持这个操作；forward_list 不支持这个操作，它有自己专门版本的 earse；
        </td>
    </tr>
    <tr>
    	<td>c.erase(b,e)</td>
        <td>
        	删除迭代器 b 和 e所指定范围内的元素，返回一个指向最后一个被删元素之后元素的迭代器；<br/>
            若 e 是尾后迭代器，则函数也返回尾后迭代器；<br/>
            array 不支持这个操作；forward_list 不支持这个操作，它有自己专门版本的 earse；
        </td>
    </tr>
    <tr>
    	<td>c.clear()</td>
        <td>删除 c 中所有元素，返回 void；</td>
    </tr>
</table>

【注1】删除元素操作会改变容器的大小，对于 `vector` 和 `string` 来说，删除元素会使删除点之后位置的迭代器、引用和指针失效，对于 `deque` 来说，删除元素会使除首尾元素之外位置的迭代器、引用和指针失效。

【注2】删除元素的操作并不检查其参数，因此调用者必须确保删除的元素存在。

##### 7.1.2.3 访问元素

<table>
    <tr>
    	<th>顺序容器访问操作</th>
        <th>说明</th>
    </tr>
	<tr>
    	<td>c.front()</td>
        <td>
            返回容器首元素的引用；若 c 为空，函数行为未定义；<br/>
            所有顺序容器都支持这个操作；
        </td>
    </tr>
    <tr>
    	<td>c.back()</td>
        <td>
            返回容器尾元素的引用；若 c 为空，函数行为未定义；<br/>
            forward_list 不支持这个操作；
        </td>
    </tr>
    <tr>
    	<td>c[n]</td>
        <td>
            返回 c 中下标为 n 的元素的引用，n 是一个 ::size_type 类型的无符号整数，若下标越界 n>=c.size()，则函数行为未定义；<br/>
            不检查下标是否合法，由调用者保证；<br/>
            只适用于 string、vector、deque 和 array；
        </td>
    </tr>
    <tr>
    	<td>c.at(n)</td>
        <td>
            返回下标为 n 的元素的引用；如果下标越界 n>=c.size()，则抛出 out_of_range 异常；<br/>
            只适用于string、vector、deque 和 array；
        </td>
    </tr>
</table>

【注1】调用 `front` 和 `back`、或解引用 `begin` 和 `end` 之前，都必须确保容器非空。

【注2】访问元素操作返回的都是引用，如果容器是一个 `const` 对象，则上述操作返回的是 `const` 引用。

##### 7.1.2.4 替换元素

<table>
    <tr>
    	<th>assign 操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>seq.assign(b,e)</td>
        <td>将 seq 中的元素替换为迭代器 b 和 e 所表示的范围中的元素。迭代器 b 和 e 不能指向 seq 中的元素</td>
    </tr>
    <tr>
    	<td>seq.assign(il)</td>
        <td>将 seq 中的元素替换为初始化列表 il 中的元素</td>
    </tr>
    <tr>
    	<td>seq.assign(n,t)</td>
        <td>将 seq 中的元素替换为 n 个值为 t 的元素</td>
    </tr>
</table>
【注1】赋值相关运算会导致指向左边容器内部的迭代器、引用和指针失效。

【注2】赋值运算符（`6.3.3 一般容器操作`）要求左右两边的运算对象具有相同的类型，顺序容器的 `assign` 成员允许从一个不同但相容的类型赋值。

【注3】由于旧元素被替换，因此传递给 `assign` 的迭代器不能指向调用 `assign` 的容器。

##### 7.1.2.5 容器大小操作

<table>
    <tr>
    	<th>顺序容器大小操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c.resize(n)</td>
        <td>
            调整 c 的大小为 n 个元素。若 n&lt;c.size()，则丢弃多余元素。如果 n&gt;c.size()，则新元素进行值初始化；<br/>
            array 不支持这个操作；<br/>
        </td>
    </tr>
    <tr>
    	<td>c.resize(n,t)</td>
        <td>
            调整 c 的大小为 n 个元素。若 n&lt;c.size()，则丢弃多余元素。如果 n&gt;c.size()，则新元素初始化为值 t；<br/>
            array 不支持这个操作；
        </td>
    </tr>
</table>
【注】如果缩小容器会改变容器大小，指向被删除元素的迭代器、引用和指针都会失效。

#### 7.1.3 容器操作与迭代器失效归纳

书本 `p315`，暂略，用到时直接查。

* 案例：编写改变容器的循环程序

  ```c++
  // 傻瓜循环，删除偶数元素，复制每个奇数元素
  vector<int> vi = {0,1,2,3,4,5,6,7,8,9};
  auto iter = vi.begin();  				// 调用 begin 而不是 cbegin，因为我们要改变 vi
  while (iter != vi.end())
  {
      if (*iter % 2) {
          // 复制当前元素
          iter = vi.insert(iter, *iter);  // 重要手法，将 *iter 元素值拷贝到 iter 位置前，并返回指向新元素的迭代器，赋给 iter
          iter += 2;  					// 重要注意，先跳过新元素，然后跳过旧元素，最后才指向了下一个元素
      }
      else
          // 删除偶数元素
          iter = vi.erase(iter); 			// 重要注意，不向前移位，因为 iter 已经指向了删除元素之后的元素
  }
  ```

* 案例：不要保存 `end` 返回的迭代器

  增删操作总是会使 `end` 返回的迭代器失效，因此循环体内必须返回调用 `end`，否则就会出错：

  ```c++
  // 错误，此循环是未定义的
  auto begin = v.begin(), end = v.end();  // 保存尾后迭代器的值是一个坏主意
  while(begin != end)  					// 建议改成 begin != v.end()，即不保存尾后迭代器
  {
      ++begin;
      begin = v.insert(begin,42);
      ++begin;
  }
  ```

<table>
    <tr>
    	<th>语法</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>C c;</td>
        <td>
            默认构造函数。<br/>
            【注】如果 C 是一个 array，则 c 中的元素按默认初始化方式初始化，否则 c 为空；
        </td>
    </tr>
    <tr>
    	<td>C c1(c2)</td>
        <td rowspan="2">
            c1 初始化为 c2 的拷贝。<br/>
    		【注1】c1 和 c2 必须是相同的（容器和元素）类型；<br/>
            【注2】对于 array 类型，两者还必须具有相同的大小；
        </td>
    </tr>
    <tr>
    	<td>C c1=c2</td>
    </tr>
    <tr>
    	<td>C c{a,b,c...}</td>
        <td rowspan="2">
            c 初始化为列表中元素的拷贝。<br/>
            【注1】列表中的元素类型必须与 C 的元素类型相容；<br/>
            【注2】对于 array 类型，列表中的元素数目必须小于等于 array 的大小，剩余元素的将进行值初始化；
        </td>
    </tr>
    <tr>
    	<td>C c={a,b,c...}</td>
    </tr>
    <tr>
    	<td>C c(b,e)</td>
        <td>
            c 初始化为迭代器 b 和 e 指定范围中的元素的拷贝；<br/>
            【注】范围中的元素的类型必须与 C 的元素类型相容（array 不适用）;
        </td>
    </tr>
    <tr>
    	<td colspan="2"><b>只有顺序容器（不包括 array）的构造函数才能接受大小参数</b></td>
    </tr>
    <tr>
    	<td>C seq(n)</td>
        <td>
            seq 包含 n 个元素，这些元素进行了值初始化；此构造函数是 explicit 的（string不适用）。<br />
            【注1】如果元素类型是内置类型或有默认构造函数的类类型，可以使用此函数；<br/>
            【注2】如果元素类型没有默认构造函数，则必须用下面的函数显示提供初值；
        </td>
    </tr>
    <tr>
    	<td>C seq(n,t)</td>
        <td>seq 包含 n 个初始化为值 t 的元素</td>
    </tr>
</table>

### 7.2 标准库类型 `string`

#### 7.2.1 `string` 的定义和初始化

一般容器的定义和初始化：`6.3.2 容器定义和初始化`；

顺序容器的定义和初始化：`6.3.2 容器定义和初始化`；

```c++
#include <string>
using std::string;

string s1;				// 默认初始化，s1 是一个空字符串

string s2(s1);			// 直接初始化，s2 是 s1 的副本
string s2 = s1; 		// 拷贝初始化，等价于 s2(s1)

string s3("value");		// 直接初始化，除最后那个空字符外其他所有字符都被拷贝
string s3 = "value";	// 拷贝初始化，除最后那个空字符外其他所有字符都被拷贝

string s4(n, 'c');		// 直接初始化，将 s4 初始化为由连续 n 个字符 c 组成的串
```

`string` 的定义和初始化：

| 语法                            | 说明                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| `string s(cp)`                  | `s` 是 `cp`（`const char*`）指向的数组中字符的拷贝。<br/>【注1】数组 `cp` 必须以空字符 `'\n'` 结尾，拷贝操作遇到它才会停止；<br/>【注2】`n` 是无符号值； |
| `string s(cp,n)`                | `s` 是 `cp` 指向的数组中前 `n` 个字符的拷贝（数组至少应该含有 `n` 个字符）。<br/>【注1】数组 `cp` 无须以空字符 `'\n'` 结尾，拷贝操作根据 `n` 停止，越界时行为是未定义的；<br/>【注2】`n` 是无符号值； |
| `string s(s2,pos)`              | `s` 是 `string s2` 从下标 `pos2` 开始的字符的拷贝。若 `pos2>s2.size()`，构造函数的行为未定义。 |
| `string s(s2,pos2,len2)`        | `s` 是 `string s2` 从下标 `pos2` 开始 `len2` 个字符的拷贝。若 `pos2>s2.size()`，构造函数的行为未定义。<br/>【注1】若 `len2` 超出 `s2` 的范围，即越界时，至多拷贝 `s2.size()-pos2` 个字符；<br/>【注2】`pos2`，`len2` 是无符号值； |
| `s.substr(pos=0)`               | 返回 `string s` 从下标 `pos` 开始字符的拷贝。若 `pos>s.size()`，抛出 `out_of_range` 异常。 |
| `s.substr(pos3=0,len3=end-pos)` | 返回 `string s` 从下标 `pos3` 开始 `len3` 个字符的拷贝。若 `pos3>s.size()`，抛出 `out_of_range` 异常。<br/>【注1】若 `len3` 超出 `s` 的范围，即越界时，至多拷贝 `s.size()-pos3` 个字符；<br/>【注2】`pos3`，`len3` 是无符号值； |

#### 7.2.2 `string` 的基本操作

一般容器的操作：`6.3.3 一般容器操作`；

顺序容器的操作：`7.1.2 顺序容器操作`；

`string` 的基本操作：

| 操作                 | 含义与要点                                                   |
| -------------------- | ------------------------------------------------------------ |
| `os << s`            | 将 `s` 写到输出流 `os` 当中，返回 `os`；                     |
|                      | 【注1】`string` 对象会自动忽略开头的空白，直到遇见下一处空白为止； |
|                      | 【注2】可以使用输入流返回的对象作为循环体判断条件来实现读取未知数量的 `string` 对象； |
| `is >> s`            | 从 `is` 中读取字符串赋给 `s`，字符串以空白分隔，返回 `is`    |
| `getline(is, s)`     | 从 `is` 中读取一行赋给 `s`，返回 `is`                        |
|                      | 【注1】`getline` 函数能够保留输入时的空白符，直到遇见换行符为止（换行符也读进来），然后存入 `string` 对象（丢弃换行符） |
| `s.empty()`          | `s` 为空返回 `true`，否则返回 `false`                        |
| `s.size()`           | 返回 `s` 中字符的个数                                        |
|                      | 【注1】`size` 函数返回的是一个 `string::size_type` 类型的值，大多数标准库类型都会定义一些配套类型 |
|                      | 【注2】可以肯定的是 `string::size_type` 是一个无符号类型的值，因此要注意 `2.3.1.2 算数类型的转换` 提到的与带符号数混用问题 |
|                      | 【注3】除了直接用 `string::size_type` 外，编译器允许通过 `auto` 或 `decltype` 来推断使用该类型 |
| `s[n]`               | 返回 `s` 中第 `n` 个字符的引用，位置 `n` 从 0 计起           |
| `s1 + s2`            | 返回 `s1` 和 `s2` 连接的结果                                 |
|                      | 【注1】可以将字符字面值和字符串字面值与 `string` 相加，不过必须确保加号两边至少有一个是 `string` 对象 |
|                      | 【注2】由于历史原因，字符串字面值类型为 `const char*`，与 `string` 是不同的类型 |
| `s1 = s2`            | 用`s2`的副本代替`s1`中原来的字符，即赋值操作                 |
| `s1 == s2`           | 如果 `s1` 和 `s2` 中所含的字符完全一样，则判断相等，否则不等，对大小写敏感 |
| `s1 != s2`           | 如果 `s1` 和 `s2` 中所含的字符完全一样，则判断相等，否则不等，对大小写敏感 |
| `<`, `<=`, `>`, `>=` | 利用字符在字典中的顺序进行比较，对大小写敏感                 |

`p327 9.5.4` 的 `compare` 函数，暂略。

`p322 9.5.2` 改变 `string` 的其他方法，暂略。

#### 7.2.3 `string` 的字符操作

在 `<cctype>` 头文件中定义了一组标准库函数来处理字符：

| 函数          | 作用                                                         |
| ------------- | ------------------------------------------------------------ |
| `isalnum(c)`  | 当 `c` 时字母或数字时为真                                    |
| `isdigit(c)`  | 当 `c` 是数字时为真                                          |
| `isxdigit(c)` | 当 `c` 是十六进制数字时为真                                  |
| `isalpha(c)`  | 当 `c` 是字母时为真                                          |
| `islower(c)`  | 当 `c` 是小写字母时为真                                      |
| `isupper(c)`  | 当 `c` 是大写字母时为真                                      |
| `tolower(c)`  | 如果 `c` 是大写字母，则输出对应的小写字母；否则原样输出      |
| `toupper(c)`  | 如果 `c` 是小写字母，则输出对应的大写字母；否则原样输出      |
| `iscntrl(c)`  | 当 `c` 是控制字符时为真                                      |
| `ispunct(c)`  | 当 `c` 是标点符号时为真（即`c`不是控制字符、数字、字母、可打印空白） |
| `isspace(c)`  | 当 `c` 是空白时为真（即`c`是空格、横纵制表符、回车符、换行符、进纸符） |
| `isgraph(c)`  | 当 `c` 不是空格但可打印时为真                                |
| `isprint(c)`  | 当 `c` 是可打印字符时为真（即 `c` 是空格或 `c` 具有可视形式） |

【注1】C++ 兼容 C 语言的标准库，C 语言头文件形如 `<name.h>`，它的 C++ 版本为 `<cname>`，其中定义的名字属于命名空间 `std`。

【注2】可以使用范围的 `for` 语句处理每个字符。

【注3】可以下标和迭代器处理 `string` 中的部分字符：

#### 7.2.4 `string` 的搜索操作

* `constexpr size_type find( const basic_string& str, size_type pos = 0 ) const noexcept;`

  * `str` - string to search for

  * `pos` - position at which to start the search
  * `return` - Position of the first character of the found substring or `string::npos` if no such substring is found

  ```c++
  std::string::size_type n;
  std::string const s = "This is a string";
   
  // search from beginning of string
  n = s.find("is");
  print(n, s);
  ```

#### 7.2.5 `string` 的数值转换

<table>
    <tr>
    	<th>转换操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>to_string(val)</td>
        <td>
            一组重载函数，返回数值 val 的 string 表示；<br/>
            val 可以是任何算数类型，包括浮点类型和 int 或更多的整型；<br/>
            与往常一样，小整型会被提升；
        </td>
    </tr>
    <tr>
    	<td>stoi(s,p,b)</td>
        <td rowspan=5>
            返回 s 的起始子串（表示整数内容）的数值，返回值类型分别是 int、long、unsigned long、long long、unsigned long long；<br/>
            b 表示转换所用的基数，默认值为 10；<br/>
            p 是 size_t 指针，用来保存 s 中第一个非数值字符的下标，p 默认为 0，即函数不保存下标；
        </td>
    </tr>
    <tr>
    	<td>stol(s,p,b)</td>
    </tr>
    <tr>
    	<td>stoul(s,p,b)</td>
    </tr>
    <tr>
    	<td>stoll(s,p,b)</td>
    </tr>
    <tr>
    	<td>stoull(s,p,b)</td>
    </tr>
    <tr>
    	<td>stof(s,p)</td>
        <td rowspan=3>
        	返回 s 的其实子串（表示浮点数内容）的数值，返回值类型分别是 float、double 或 long double；<br/>
            参数 p 的作用于整数转换含中一样；
        </td>
    </tr>
    <tr>
    	<td>stod(s,p)</td>
    </tr>
    <tr>
    	<td>stold(s,p)</td>
    </tr>
</table>
【注1】`string` 中第一个非空白字符：

* 必须是符号（ `+` 或 `-` ）或基数数字范围内的数字
* 对于浮点数，也可以以小数点（`.`）开头，并可以包含 `e` 或 `E` 表示指数部分

【注2】`string` 转换中的异常：

* 如果 `string` 不能转换为一个数值，则抛出 `invalid_argument` 异常（`3.6.3 标准异常`）
* 如果目标数值超出范围，则抛出 `out_of_range` 异常（`3.6.3 标准异常`）

### 7.3 标准库类型 `vector`

#### 7.3.1 `vector` 的定义和初始化

一般容器的定义和初始化：`6.3.2 容器定义和初始化`；

顺序容器的定义和初始化：`6.3.2 容器定义和初始化`；

```c++
#include <vector>
vector<T> v1;				// 默认初始化，v1 是一个空 vector，包含 T 类型的元素，这些元素执行默认初始化，因此对元素有要求

vector<T> v2(v1);			// 直接初始化，v2 中包含了 v1 所有元素的副本
vector<T> v2 = v1;			// 直接初始化，v2 中包含了 v1 所有元素的副本

vector<T> v3(n, val);		// 直接初始化，v3 包含了 n 个重复的元素，每个元素值为 val
vector<T> v4(n);			// 直接初始化，v4 包含了 n 个重复的元素，每个元素执行默认初始化，因此对元素有要求

vector<T> v5{a,b,c...}		// 列表初始化，v5 包含了列表中的元素
vector<T> v5 = {a,b,c...}	// 列表初始化，v5 包含了列表中的元素
```

【注】`vector` 的接受大小的单参构造函数是 `explicit` 的。

```c++
vector<int> v1(10);				// 正确：直接初始化
vector<int> v2 = 10;			// 错误：接受大小参数的构造函数是 explicit 的

void f(vector<int>);			// 定义 f 函数，它的形参进行拷贝初始化（1.2.1.2 变量初始化）
f(10);							// 错误：不能用一个 explicit 的构造函数拷贝一个实参
f(vector<int>(10));				// 正确：从一个 int 直接构造一个临时 vector
```

【辨】直接初始化与列表初始化的含义，可以通过使用花括号或圆括号来区分：

* 花括号里的都是元素，圆括号里的是参数，一个为数量，两个为数量和值

* 对于花括号来说，当元素刁钻使得列表初始化无法进行时，会考虑使用其他的初始化方式

```c++
vector<int> v1(10);				// v1 有 10个元素，每个的值都是 0
vector<int> v2{10};				// v2 有 1 个元素，值是 10

vector<int> v3(10, 1);			// v3 有 10 个元素，每个的值都是 1
vector<int> v4{10, 1};			// v4 有 2 个元素，值分别是 10 和 1

vector<string> v5{"hi"};		// 列表初始化，v5 有一个 string 元素
vector<string> v6("hi");		// 错误，不能使用字符串字面值构建 vector 对象

vector<string> v7{10};			// 直接初始化，因为 10 不是 string，不能执行列表初始化，最终有 10 个空 string 元素
vector<string> v8{10, "hi"};	// 直接初始化，因为 10 不是 string，不能执行列表初始化，最终有 10 个 "hi" 元素
```

#### 7.3.2 `vector` 的基本操作

一般容器的操作：`6.3.3 一般容器操作`；

顺序容器的操作：`7.1.2 顺序容器操作`；

`vector` 的基本操作：

| 操作                 | 含义与要点                                                   |
| -------------------- | ------------------------------------------------------------ |
| `v.empty()`          | 如果 `v` 不含任何元素，返回 `true`；否则返回 `false`         |
| `v.size()`           | 返回 `v` 中元素的个数                                        |
|                      | 【注1】`size` 函数返回的是一个 `vector<T>::size_type` 类型的值 |
| `v.push_back(t)`     | 向 `v` 的尾端添加一个值为 `t` 的元素                         |
|                      | 【注1】`vector` 能够高效的增长，除非所有元素值都一样，否则不必设定对象大小 |
|                      | 【注2】如果循环体内部包含有向 `vector` 对象添加元素的语句，则不能使用范围 `for` 循环，原因见 `3.4.3 范围 `for` 语句` |
| `v[n]`               | 返回 `v` 中第 `n` 个位置上的引用，位置 `n` 从0计起           |
|                      | 【注1】可以用下标访问/修改已经存在的元素，但不能用于添加元素 |
|                      | 【注2】确保下标合法的有效手段是使用范围 `for`语句            |
| `v1 = v2`            | 用 `v2` 中元素的拷贝替换 `v1` 中的元素                       |
| `v1 = {a,b,c,...}`   | 用列表中元素的拷贝替换 `v1` 中的元素                         |
| `v1 == v2`           | `v1` 和 `v2` 相等当且仅当它们的元素数量相等、对应位置的元素值相同 |
| `v1 != v2`           | `v1` 和 `v2` 相等当且仅当它们的元素数量相等、对应位置的元素值相同 |
| `<`, `<=`, `>`, `>=` | 以字典顺序进行比较                                           |

#### 7.3.3 `vector` 的==增长过程（详细策略）==

尽管容器的使用与实现分离，但对于 `vector` 和 `string` 来说，其部分实现渗透到了接口当中。

为了减少这两种顺序容器空间变化的开销，标准库采用可以减少容器空间重新分配次数的策略：

* 标准库通常会为容器分配比实际需求更大的空间
* 只有迫不得已时才会分配新的内存空间

因此，也相应配有额外的容器大小管理操作。

<table>
    <tr>
    	<th>容器大小管理操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c.capacity()</td>
        <td>
            不重新分配内存空间的话，c 可以保存多少元素；<br/>
            只适用于 vector 和 string；
        </td>
    </tr>
    <tr>
    	<td>c.reserve(n)</td>
        <td>
            分配至少能容纳 n 个元素的内存空间，n 大只增，n 小不减；<br/>
            只适用于 vector 和 string；
        </td>
    </tr>
    <tr>
    	<td>c.shrink_to_fit()</td>
        <td>
            请求将 capacity() 减少为与 size() 相同的大小，仅仅是请求而已；<br/>
            只适用于 vector、string 和 deque；
        </td>
    </tr>
</table>
### 7.4 标准库类型 `array`

#### 7.4.1 定义和初始化

定义标准库 `array` 时，必须同时指定元素类型和大小：

```c++
// array 类型定义
array<int, 42> a;				// 类型为：保存 42 个 int 的数组
array<string, 10> b;			// 类型为：保存 10 个 string 的数组
    
// array 相关类型定义
array<int, 10>::sizetype i;		// 正确
array<int>::sizetype;			// 错误：array<int> 不是一个类型
```

【注1】`array` 大小固定的特性，影响它所定义的构造函数的行为：

* 默认初始化 `array` 时，包含与其大小一样多的元素，里面的元素都被默认初始化
* 值初始化 `array` 时，先用初始值直接初始化靠前元素，剩余的元素都被值初始化
* 如果元素类型是类类型，则该元素类型必须有一个默认构造函数，以便值初始化能够进行

【注2】与其他容器不同，一个默认构造的 `array` 是非空的：

* 数组特性：`array` 的初始化和内置数组一样，拥有默认初始化和值初始化特性

* 容器特性：不同于内置数组，`array` 可以进行拷贝或对象赋值操作

【悟】`array` 是一个改进的、对象型的数组。

### 7.5 容器适配器

#### 7.5.1 容器适配器概述

适配器是标准库中的一个通用概念，容器、迭代器和函数都有适配器。

本质上，一个适配器是一种机制，基于某种功能对其进行“装饰”，得到另外一种功能，和”装饰模式“的思想类似。

对于顺序容器，标准库定义了三个顺序容器适配器：

| 顺序容器适配器   | 说明     |
| ---------------- | -------- |
| `stack`          | 栈       |
| `queue`          | 队列     |
| `priority_queue` | 优先队列 |

【注】栈和队列（堆）底层都是基于 `deque` 实现的，它最通用。

#### 7.5.2 容器适配器的定义和初始化

每个适配器都定义了两个构造函数：

* 默认构造函数，创建一个空对象
* 接受一个容器、迭代器或函数的构造函数，通过拷贝目标来创建和初始化容器适配器

对于顺序容器，它们的定义语法为：

```c++
// 适配器名<元素类型, 容器类型> 变量名;
stack<string, vector<string>> str_stk;  						// 其中容器类型可选，默认 vector
// 配器名<元素类型, 容器类型> 变量名(容器对象);
stack<string, vector<string>> str_stk2(svec);  					// 其中容器类型可选，默认 vector
// 配器名<元素类型, 容器类型, 排序方式> 变量名(容器对象); 
priority_queue<int, vector<int>, greater<int>> int_stk(ivec); 	// 其中元素排序方式可选，默认 < 为最大堆
```

#### 7.5.3 容器适配器的基本操作

每个容器适配器都基于底层容器类型定义了自己的特殊操作，但不可以直接使用底层容器类型的操作。

<table>
    <tr>
        <th>通用操作和类型</th>
        <th>说明</th>
    </tr>
    <tr>
        <td>size_type</td>
		<td>一种类型，足以保存当前的最大对象的大小</td>
    </tr>
    <tr>
    	<td>value_type</td>
        <td>元素类型</td>
    </tr>
    <tr>
    	<td>container_type</td>
        <td>实现适配器的底层容器类型</td>
    </tr>
    <tr>
    	<td>A a;</td>
        <td>创建一个名为 a 的适配器</td>
    </tr>
    <tr>
    	<td>A a(c);</td>
		<td>创建一个名为 a 的适配器，带有容器 c 的一个拷贝</td>
    </tr>
    <tr>
    	<td>关系运算符</td>
        <td>每个适配器都支持所有关系运算符：==、!=、&lt;、&lt;=、&gt;、&gt;=</td>
    </tr>
    <tr>
    	<td>a.empty()</td>
        <td>若 a 包含任何元素，返回 false，否则返回 true</td>
    </tr>
    <tr>
    	<td>a.size()</td>
        <td>返回 a 中包含的元素数目</td>
    </tr>
    <tr>
    	<td>swap(a,b)</td>
        <td rowspan=2>交换 a 和 b 的内容，a 和 b 必须有相同类型，包括底层容器类型也必须相同</td>
    </tr>
    <tr>
    	<td>a.swap(b)</td>
    </tr>
</table>

#### 7.5.4 栈的基本操作

<table>
    <tr>
    	<th>栈的基本操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td colspan=2>
        	stack 类型定义在 &lt;stack&gt; 头文件中；<br/>
            栈默认基于 deque 实现，也可以在 list 或 vector 之上实现；
        </td>
    </tr>
    <tr>
    	<td>s.pop()</td>
        <td>删除栈顶元素，但不返回该元素的值</td>
    </tr>
    <tr>
    	<td>s.push(item)</td>
        <td>创建一个新元素压入栈顶，该元素通过拷贝或移动 item 而来</td>
    </tr>
    <tr>
    	<td>s.emplace(args)</td>
        <td>由 args 构造元素压入栈顶</td>
    </tr>
    <tr>
    	<td>s.top()</td>
        <td>返回栈顶元素，但不将元素弹栈</td>
    </tr>
</table>
#### 7.5.5 队列的基本操作

<table>
    <tr>
    	<th>队列的基本操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td colspan=2>
        	queue 和 priority_queue 类型定义在 &lt;queue&gt; 头文件中；<br/>
            queue 默认基于 deque 实现，priority_queue 默认基于 vector 实现；<br/>
            queue 可以用 list 或 vector 实现，priority_queue 可以用 deque 实现；
        </td>
    </tr>
    <tr>
    	<td>q.pop()</td>
        <td>返回 queue 的首元素或 priority_queue 的最高优先级元素</td>
    </tr>
    <tr>
    	<td>q.front()</td>
        <td rowspan=2>
            返回首元素或尾元素，但不删除此元素；<br/>
            只适用于 queue；
        </td>
    </tr>
    <tr>
    	<td>q.back()</td>
    </tr>
    <tr>
    	<td>q.top()</td>
        <td>
            返回最高优先级元素，但不删除该元素；<br/>
            只适用于 priority_queue；
        </td>
    </tr>
	<tr>
		<td>q.push(item)</td>
        <td>在 queue 末尾或 priority_queue 中恰当位置创建一个元素</td>
	</tr>
    <tr>
    	<td>q.empalce(args)</td>
        <td>由 args 构造一个元素进队</td>
    </tr>
</table>
【注1】默认情况下，标准库在元素类型上使用 `<` 元素符来确定其在 `priority_queue` 中的优先级，是一个大顶堆。

【注2】队列跟排队一样对头（front）出，队尾（back）入，C++ 向量中，`[back, ..., front]`。

【注3】优先队列自定义比较操作，第三个模板参数是类型，要求是一种可调用对象（`13.4 可调用对象`）。

* 方法一：重载 `less<T>` 或 `greater<T>` 的比较运算符（`13.3 标准库运算符对象`）

  ```c++
  // 1 结构体
  struct A
  {
      int l;
      int r;
      int label;
  }a;
  
  // 2 重载比较运算符
  // less<T> 使用
  bool operator < (A a1, A a2){
  	return a1.r < a2.r;
  }
  // greater<T> 使用
  bool operator > (A a1, A a2){
  	return a1.l > a2.l;
  }
  
  // 3 定义优先队列
  priority_queue<a, vector<a>, greater<a> > que1;
  priority_queue<a, vector<a>, less<a> > que2;
  ```

* 方法二：重写仿函数（即函数对象，`13.2.7 函数调用运算符`）的调用运算符

  ```c++
  struct cmp{
  	bool operator()(Node a, Node b){
  		//a < b时为降序，a > b时为升序，此时为降序
  		if(a.x == b.x) return a.y < b.y;
  		return a.x < b.x;
  	}
  };
  
  // 此时为大顶堆，元素降序，注意 cmp 使用的方式，不带"()"
  priority_queue<Node, vector<Node>, cmp> q2;
  ```

* 方法三：自定义函数指针

  ```c++
  static bool cmp(pair<int, int>& m, pair<int, int>& n) {
          return m.second > n.second;
  }
  
  priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(&cmp)> q(cmp); // 正确：类模板实参为可调用对象的类型
  // 【错误：“4.5.2 函数指针形参” 的用值特性在传值过程中体现，但这里是模板参数类型传递，不是普通参数值传递；
  //      decltype(cmp) 搭配 * 的形式，并不是一种声明类型的方式，它只是声明变量的方式；】【保留意见】
  priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)*> q(cmp); // 正确：类模板实参为可调用对象的类型
  
  q1.push({0,1});
  q1.push({1,2});
  
  q2.push({0,1});
  q2.push({1,2});
  ```
  
  【注】这种场景，类似于 `unique_ptr` 的 `11.3.3.1 定义和初始化`，原理详见 `15.4.3 效率灵活设计` 节。
