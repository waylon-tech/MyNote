### 目录

[toc]

### 8.1 关联容器

#### 8.1.1 关联容器概述

<table>
    <tr>
    	<th>关联容器类型</th>
        <th>说明</th>
    </tr>
    <tr>
        <th colspan=2>有序集合（底层红黑树）</th>
    </tr>
    <tr>
    	<td>map</td>
        <td>关联数组；保存“关键字-值”对</td>
    </tr>
    <tr>
    	<td>set</td>
        <td>只保存关键字（即值）的数组</td>
    </tr>
    <tr>
    	<td>multimap</td>
        <td>关键词可重复出现的 map</td>
    </tr>
    <tr>
    	<td>multiset</td>
        <td>关键词可重复出现的 set</td>
    </tr>
    <tr>
    	<th colspan=2>无序集合（底层哈希表）</th>
    </tr>
    <tr>
    	<td>unordered_map</td>
        <td>用哈希函数组织的 map</td>
    </tr>
    <tr>
    	<td>unordered_set</td>
        <td>用哈希函数组织的 set</td>
    </tr>
    <tr>
    	<td>unordered_multimap</td>
        <td>用哈希函数组织的 map，关键字可以重复出现</td>
    </tr>
    <tr>
    	<td>unordered_multiset</td>
        <td>用哈希函数组织的 set，关键字可以重复出现</td>
    </tr>
</table>

【注1】每个容器都定义在一个头文件中：

* `map` 和 `multimap` 定义在头文件 `<map>` 中
* `set` 和 `multiset` 定义在头文件 `<set>` 中
* `unordered_map` 和 `unordered_multimap` 定义在头文件 `<unordered_map>` 中
* `unordered_set` 和 `unordered_multiset` 定义在头文件 `<unordered_set>` 中

【注2】这 8 个容器间的不同体现在三个维度上：

* 或者是一个`set`，或者是一个`map`
* 或者不要求重复的关键字，或者允许重复关键字
* 或者按顺序保存元素，或者无序保存

#### 8.1.2 关联容器定义和初始化

##### 8.1.2.1 定义和初始化语法

一般容器的定义和初始化：`6.3.2 容器定义和初始化`；

关联容器的类型需要补充元素类型：

* 定义一个 `map` 时，需要指出关键字类型和值类型
* 定义一个 `set` 时，需要指出关键字类型

每个关联容器都定义了三个构造函数：

* 默认构造函数，创建一个指定类型的空容器
* 接受一个容器、迭代器或函数的构造函数，通过拷贝目标来创建和初始化关联容器
* 接受一个列表的构造函数，进行列表初始化

```c++
map<string, size_t> word_count;				// 默认初始化
map<string, string> authors = {				// 列表初始化
    {"Joyce", "James"},
    {"Austen", "Jane"},
};
										
set<int> iset(iset2);						// 直接初始化
set<int> iset(ivec.cbegin(), ivec.cend());	// 直接初始化
set<string> exclude = {"a", "an", "the"};	// 列表初始化
```

##### 8.1.2.2 关键字类型的要求

**有序容器的关键字类型要求**

对于有序容器 `map`、`set`、`multimap` 和 `multiset`：

* 关键字类型必须定义元素比较的方法，默认情况下，标准库使用关键字类型的 `<` 运算符

可以向算法提供自定义的比较操作，代替关键字上的 `<` 运算符：

* 自定义操作的要求：严格弱序性质
  * 两个关键字不能同时“小于等于”对方
  * 关键字的“小于等于”序可以传递
  * 如果两个关键字都不“小于等于”另一个，则称这两个关键字等价，这个等价要能传递

* 自定义操作的使用：传入额外参数
  * 定义了符合严格弱序性质的操作后，要在关联容器模板参数传入函数指针的类型、构造函数参数传入函数指针

```c++
bool compareIsbn(const Sales_data &lhs, const Sales_data &rhs)
{
    return lhs.isbn() < rhs.isbn();
}

// 参数是比较操作指针
multiset<Sales_data, decltype(compareIsbn)*> bookstore(compareIsbn);
```

**无序容器的关键字类型要求**

对于无序容器 `unordered_map`、`unordered_set`、`unordered_multimap` 和 `unordered_multiset`：

* 关键字类型必须定义元素相等的方法，默认情况下，标准库使用关键字类型的 `==` 运算符
* 容器同时使用 `hash<key_type>` 类型的对象来生成每个元素的哈希值
  * 标准库为内置类型（包括指针）、`string` 和智能指针提供了 `hast<key_type>` 模板，可以直接定义这些类型的无序容器
  * 不能直接定义关键字为自定义类类型的无序容器，必须或隐或显地（定义给类或直接传入）提供一个 `hash` 函数（==16.5==）

案例演示：显示提供 `hash` 函数

```c++
size_t hasher(const Sales_data &sd)
{
    return hash<string>() (sd.isbn());
}
bool eqOp(const Sales_data &lhs, const Sales_data &rhs)
{
    return lhs.isbn() == rhs.isbn();
}

using SD_multiset = unordered_multiset<Sales_data, decltype(hasher)*, decltype(eqOp)*>;
// 参数是：容器大小、哈希函数指针、相等性判断函数指针
SD_multiset bookstroe(42, hasher, eqOp);
```

##### 8.1.2.3 pair 类型

当从 `map` 类容器提取一个元素时，得到一个 `pair` 类型的对象，`pair` 类型是关联容器的中间类型，定义在头文件 `<utility>` 中。

`pair` 类型的操作如下：

<table>
    <tr>
    	<th>pair 上的操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>pair&lt;T1, T2&gt; p;</td>
        <td>p 是一个 pair，两个类型分别为 T1 和 T2 的成员都进行了值初始化</td>
    </tr>
    <tr>
    	<td rowspan=2>pair&lt;T1, T2&gt; p(v1, v2);</td>
        <td>p 是一个成员类型为 T1 和 T2 的 pair；first 和 second 成员分别用 v1 和 v2 初始化</td>
    </tr>
    <tr>
    	<td>等价于 pair&lt;T1, T2&gt; p = {v1, v2};</td>
    </tr>
    <tr>
    	<td>make_pair(v1, v2)</td>
        <td>返回一个用 v1 和 v2 初始化的 pair。pair 的类型从 v1 和 v2 的类型推断出来</td>
    </tr>
    <tr>
    	<td>p.first</td>
        <td>返回名为 first 的（公有）数据成员</td>
    </tr>
    <tr>
    	<td>p.second</td>
        <td>返回名为 second 的（公有）数据成员</td>
    </tr>
    <tr>
        <td>p1 relop p2</td>
        <td>关系运算符（&lt;、&gt;、&lt;=、&gt;=）按字典序定义（类似于 pareto），基于元素的 &lt; 运算符实现；<br/>
            当 p1.first &lt; p2.first || ( !(p2.first &lt; p1.first) &amp; p1.second &lt; p2.second ) 时，p1 &lt; p2 为 true；
        </td>
    </tr>
    <tr>
    	<td>p1 == p2</td>
        <td rowspan=2>当 first 和 second 成员分别相等时，两个 pair 相等。基于元素的 &lt; 运算符实现</td>
    </tr>
    <tr>
    	<td>p1 != p2</td>
    </tr>
</table>

#### 8.1.3 关联容器操作

一般容器的操作：`6.3.3 一般容器操作`；

此外，除了顺序容器的位置相关操作，关联容器支持`p305 9.3`节中所述的<u>顺序容器操作</u>。

##### 8.1.3.1 类型别名

关联容器还有额外的类型别名：

<table>
    <tr>
    	<th>关联容器类型别名</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>key_type</td>
        <td>此容器类型的关键字类型</td>
    </tr>
    <tr>
    	<td>mapped_type</td>
        <td>每个关键字关联的类型，只适用于 map</td>
    </tr>
    <tr>
    	<td>value_type</td>
        <td>
            对于 set，与 key_type 相同；<br/>
            对于 map，为 pair&lt;const key_type, mapped_type&gt;，关键字部分是 const 的；
        </td>
    </tr>
</table>

##### 8.1.3.2 迭代器

关联容器迭代器和顺序容器迭代器一样使用，只不过还需注意：

* 当解引用一个关联容器迭代器时，会得到一个 `value_type` 的值的引用
* `set` 的迭代器是 `const` 的，无论迭代器类型是 `iterator` 还是 `const iterator`
* 通常不对关联容器使用泛型算法，常使用关联容器自身的算法

* 关联容器的迭代器都是双向的

##### 8.1.3.3 添加元素

<table>
    <tr>
    	<th>关联容器插入操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c.insert(v)</td>
        <td>v 是 value_type 类型的对象；args 用来构造一个元素</td>
    </tr>
    <tr>
    	<td>c.emplace(args)</td>
        <td>
            对于 map 和 set，只有当元素的关键字不在 c 中时才插入（或构造）元素；函数返回一个 pair，包含一个迭代器，指向具有指定关键字的元素，以及一个指示插入是否成功的 bool 值；<br/>
            对于 multimap 和 multiset，总会插入（或构造）给定元素，并返回一个指向新元素的迭代器；
        </td>
    </tr>
    <tr>
    	<td>c.insert(b,e)</td>
        <td rowspan=2>
        	b 和 e 是迭代器，表示一个 c::value_type 类型值的范围；il 是这种值的花括号列表；函数返回 void；<br/>
            对于 map 和 set，只插入关键字不在 c 中的元素；对于 multimap 和 multiset，则会插入范围中的每个元素；
        </td>
    </tr>
    <tr>
    	<td>c.insert(il)</td>
    </tr>
    <tr>
        <td>c.insert(p, v)</td>
    	<td rowspan=2>
            类似 insert(v) 与 emplace(args)，但迭代器 p 作为一个提示，指出从哪里开始搜索新元素应该存储的位置；<br/>
            返回一个迭代器，指向具有给定关键字的元素
        </td>
    </tr>
    <tr>
    	<td>c.emplace(p, args)</td>
    </tr>
</table>

【注】向 set 添加 `value_type`/`key_type` 元素比较简单，但是向 `map` 要添加的 `value_type` 元素是 `pair` 类型，所以要专门说一下

```c++
// 向 map 添加 pair 的 4 种方法
word_count.insert({word,1});								// 花括号构造传入
word_count.insert(make_pair(word,1));						// make_pair 函数构造传入
word_count.insert(pair<string, size_t>(word,1));			// pair 类型构造传入
word_count.insert(map<string, siez_t>::value_type(word,1));	// value_type 类型构造传入
```

##### 8.1.3.4 删除元素

<table>
    <tr>
    	<th>关联容器删除操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c.erase(k)</td>
        <td>从 c 中删除每个关键字（value_type 类型）为 k 的元素，返回一个 size_type 值，指出删除的元素的数量；</td>
    </tr>
    <tr>
    	<td>c.erase(p)</td>
        <td>
            从 c 中删除迭代器 p 指定的元素，p 必须指向 c 中的一个真实元素，不能等于 c.end()；<br/>
            返回一个指向 p 之后元素的迭代器，若 p 指向 c 中的尾元素，则返回 c.end()；
        </td>
    </tr>
    <tr>
    	<td>c.erase(b, e)</td>
        <td>删除迭代器对 b 和 e 所表示的范围中的元素，返回 e；</td>
    </tr>
</table>

##### 8.1.3.5 访问元素

`map` 和 `unordered_map` 容器提供了下标运算和一个对应的 at 函数，`multimap` 和 `set` 类型不支持下标操作，因为值重复或没有值。

<table>
    <tr>
    	<th>关联容器访问操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c[k]</td>
        <td>
            返回关键字为 k 的元素（mapped_type 类型）；<br/>
            如果 k 不在 c 中，添加一个关键字为 k 的元素，对其进行值初始化；
        </td>
    </tr>
    <tr>
    	<td>c.at(k)</td>
        <td>
        	访问关键字为 k 的元素，带参数检查；<br/>
            如果 k 不在 c 中，抛出一个 out_of_range 异常；
        </td>
    </tr>
</table>

【注1】与通常情况不同：

* 对一个 `map` 进行下标操作，获得 `mapped_type` 对象
* 对一个 `map` 迭代器进行解引用操作，获得 `value_type` 对象

【注2】与通常情况相同：

* `map` 下标运算返回一个左值

【注3】下标和 `at` 操作只适用于非 `const` 的 `map` 和 `unordered_map`，以应对可能的修改行为。

【注4】C++18 有一个称为 “结构化绑定” 的迭代方式：

```c++
map<int, vector<int>> mp;
// mp 的一些赋值操作
for (auto& [_, vec] : mp)
{
    // 代码
}
```

##### 8.1.3.6 查找元素

<table>
    <tr>
    	<th>关联容器查找操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td>c.find(k)</td>
        <td>返回一个迭代器，指向第一个关键字为 k 的元素，若 k 不在容器中，则返回尾后迭代器；</td>
    </tr>
    <tr>
    	<td>c.count(k)</td>
        <td>返回关键字等于 k 的元素的数量。对于不允许重复关键字的容器，返回值永远是 0 或 1；</td>
    </tr>
    <tr>
    	<td colspan=2>lower_bound、upper_bound 和 equal_bound 不适用于无序容器；</td>
    </tr>
    <tr>
    	<td>c.lower_bound(k)</td>
        <td>返回一个迭代器，指向第一个关键字不小于（&lt;=）k 的元素；</td>
    </tr>
    <tr>
    	<td>c.upper_bound(k)</td>
        <td>返回一个迭代器，指向第一个关键字大于（&gt;）k 的元素；</td>
    </tr>
    <tr>
    	<td>c.equal_range(k)</td>
        <td>返回一个迭代器 pair，表示关键字等于 k 的元素的范围。若 k 不存在，pair 的两个成员均等于 c.end()；</td>
    </tr>
</table>

【注1】在不想要下标的“自动添加”操作时，建议对 `map` 使用 `find` 代替下标访问操作。

【注2】在 `multimap` 或 `multiset` 中查找元素

如果一个 `multimap` 或 `multiset` 中有多个元素的关键字相同，则这些元素在容器中会<u>相邻存储</u>。

查找示例一：使用 `find` 和 `count`

```c++
string search_item("Alain de Botton");		// 要查找的作者
auto entries = authors.count(search_item);	// 目标元素的数量
auto iter = authors.find(search_item);		// 此作者的第一本书
// 用一个循环查找此作者的所有著作
while(entries)
{
	cout << iter->second << endl;			// 打印
    ++iter;									// 前进
    --entries;								// 记录
}
```

查找示例二：使用 `lower_bound` 和 `upper_bound`

```c++
// authors 和 search_item 的定义，与前面的程序一样
// beg 和 end 表示对应此作者的元素的范围
for(auto beg=authors.lower_bound(search_item), end=authors.upper_bound(search_item); beg!=end; ++beg)
{
    cout << beg->second << endl;	// 打印
}
```

查找示例三：使用 `equal_range`

```c++
// authors 和 search_item 的定义，与前面的程序一样
// pos 保存迭代器对，表示与关键字匹配的元素范围
for(auto pos=authors.equal_range(serach_item); pos.first!=pos.second; ++pos.first)
    cout << pos.first->second << endl;	// 打印
```

#### 8.1.4 无序容器补充

前面（`8.1.2.2 关键字类型的要求`）知道，无序容器不使用比较运算符来组织元素，而是使用一个哈希函数和关键字类型的 `==` 运算符。

除了哈希管理操作，无序容器提供了与有序容器相同的操作，两者可以相互替换使用。

**管理桶**

无序容器在存储上组织为一组桶，每个桶保存零个或多个元素，使用<u>哈希函数</u>将元素映射到桶，在桶内是<u>顺序搜索</u>元素的。

因此，无序容器的性能依赖于哈希函数的质量和桶的数量和大小。

无序容器提供了一组管理桶的函数：

<table>
    <tr>
    	<th>无序容器桶管理操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<th colspan=2>桶接口</th>
    </tr>
    <tr>
    	<td>c.bucket_count()</td>
        <td>正在使用的桶的数目</td>
    </tr>
    <tr>
    	<td>c.max_bucket_count()</td>
        <td>容器能容纳的最多的桶的数量</td>
    </tr>
    <tr>
    	<td>c.bucket_size(n)</td>
        <td>第 n 个桶中有多少个元素</td>
    </tr>
    <tr>
    	<td>c.bucket(k)</td>
        <td>关键字为 k 的元素在哪个桶中</td>
    </tr>
    <tr>
        <th colspan=2>桶迭代</th>
    </tr>
    <tr>
    	<td>local_iterator</td>
        <td>可以用来访问桶中元素的迭代器类型</td>
    </tr>
    <tr>
    	<td>const_local_iterator</td>
        <td>桶迭代器的 const 版本</td>
    </tr>
    <tr>
    	<td>c.begin(n), c.end(n)</td>
        <td>桶 n 的首元素迭代器和尾后迭代器</td>
    </tr>
    <tr>
    	<td>c.cbegin(n), c.cend(n)</td>
        <td>与前两个函数类似，但返回 const_local_iterator</td>
    </tr>
    <tr>
        <th colspan=2>哈希策略</th>
    </tr>
    <tr>
    	<td>c.load_factor()</td>
        <td>每个桶的平均元素数量，返回 float 值</td>
    </tr>
    <tr>
    	<td>c.max_load_factor()</td>
        <td>c 试图维护的平均桶大小，返回 float 值。c 会在需要时添加新的桶，以使得 load_factor&lt;=max_load_factor</td>
    </tr>
    <tr>
    	<td>c.rehash(n)</td>
        <td>重组存储，使得 bucket_count>=n 且 bucket_count>size/max_load_factor</td>
    </tr>
    <tr>
    	<td>c.reserve(n)</td>
        <td>重组存储，使得 c 可以保存 n 个元素且不必 rehash</td>
    </tr>
</table>
