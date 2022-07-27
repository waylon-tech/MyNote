### 目录

[toc]

### 9.1 泛型算法

#### 9.1.1 泛型算法概述

大多数泛型算法定义在头文件 `<algorithm>` 中，标准库还在头文件 `<numeric>` 中定义了一组数值泛型算法。

**泛型算法的特点：**

* 一般情况下，泛型算法不直接操作容器，而是遍历由两个迭代器指定的一个元素范围来进行操作

* 迭代器令泛型算法不依赖于容器，但泛型算法仍依赖于元素类型的操作

* 泛型算法永远不会执行容器的操作，从而也不会改变底层容器的大小

**泛型算法的谓词参数：**

标准库允许我们提供自定义函数来代替默认操作，通过对应重载算法新增的参数传入，称为谓词参数。

* 谓词是一个可调用的表达式，其返回结果是一个能用做条件的值
* 接受谓词参数的算法对输入序列中的元素调用谓词（函数），通过返回的条件值确定比较结果

标准库中的谓词分两种：一元谓词（只接受一个参数）和二元谓词（只接受两个参数）。

案例演示：

```c++
// 比较函数，用来按长度排序单词
bool isShorter(const string &s1, const string &s2)
{
    return s1.size() < s2.size();
}

// 按长度由短至长排序 words
stable_sort(words.begin(), words.end(), isShorter);  // 调用相应的谓词版本，最后一个参数是谓词参数，传入比较函数
```

#### 9.1.2 泛型算法结构

##### 9.1.2.1 算法分类

* 分类一：按所要求的迭代器操作分类（`6.4.1 迭代器基本概念`）

* 分类二：按照是否只读、只写或是重排序列中的元素来分类


##### 9.1.2.2 算法参数规范

大多数泛型算法具有如下 4 种形式之一：

* `alg(beg, end, other args)`
* `alg(beg, end, dest, other args)`
  * 算法假定，不管写入多少个元素到目的位置迭代器 `dest`，都是安全的
* `alg(beg, end, beg2, other args)`
  * 算法假定，`beg2` 开始的序列至少与序列 `beg` 到 `end` 一样大
* `alg(beg, end, beg2, end2, other args)`

##### 9.1.2.3 算法命名规范

* 一些算法通常有重载形式的版本，接受一个谓词参数作为新增参数（`9.1.1 泛型算法概述`）

  ```c++
  unique(beg, end);		 		// 使用 == 运算符比较元素
  unique(beg, end, comp);	 		// 使用 comp 比较元素
  ```

* 接受一个元素值的算法通常有不同名（`_if`）的版本，接受一个谓词参数代替元素值

  ```c++
  find(beg, end, val);	 		// 查找输入范围中 val 第一次出现的位置
  find_if(beg, end, pred);		// 查找第一个令 pred 为真的元素
  ```

* 变动元素的算法还提供不同名的（`_copy`）版本，用于建立对象的拷贝

  ```c++
  reverse(beg, end);				// 反转输入范围中元素的顺序
  reverse_copy(beg, end, dest);	// 将元素按逆序拷贝到 dest
  ```

#### 9.1.3 特定容器算法

与其他容器不同，链表类型 `list` 和 `forward_list` 定义了几个成员函数形式的链表版本泛型算法，如下表。

个中原因是，

* 通用版本要求随机访问迭代器，要么不适用与链表类型的迭代器，要么代价太高
* 链表版本会改变底层的容器结构，通用版本不会改变底层容器结构

<table>
    <tr>
    	<th>链表版本操作</th>
        <th>说明（以下操作都返回 void）</th>
    </tr>
    <tr>
    	<td rowspan=2>lst.merge(lst2)</td>
        <td>
            将来自 lst2 的元素合并入 lst。lst 和 lst2 都必须是有序的；<br/>
            元素将从 lst2 中删除.在合并之后，lst2 变为空；<br/>
            第一个版本使用 &lt; 运算符；第二个版本使用给定的比较操作；
        </td>
    </tr>
    <tr>
    	<td>等价于 lst.merge(lst2, comp)；</td>
    </tr>
    <tr>
    	<td>lst.remove(val)</td>
        <td rowspan=2>调用 erase 删除掉与给定值相等（==）或令一元谓词为真的每个元素；</td>
    </tr>
    <tr>
    	<td>lst.remove_if(pred)</td>
    </tr>
    <tr>
    	<td>lst.reverse()</td>
        <td>反转 lst 中元素的顺序；</td>
    </tr>
    <tr>
    	<td>lst.sort()</td>
		<td rowspan=2>使用 &lt; 或给定比较操作排序元素；</td>
    </tr>
    <tr>
    	<td>lst.sort(comp)</td>
    </tr>
    <tr>
    	<td>lst.unique()</td>
        <td rowspan=2>调用 erase 删除同一个值的连续拷贝。第一个版本使用 ==；第二个版本使用给定的二元谓词；</td>
    </tr>
    <tr>
    	<td>等价于 lst.unique(pred)；</td>
    </tr>
</table>
链表类型还定义了 `splice` 算法，这是链表数据结构所特有的，因此只有链表版本，无通用版本：

<table>
    <tr>
    	<th>链表特有操作</th>
        <th>说明</th>
    </tr>
    <tr>
    	<td colspan=2>lst.splice(args) 或 flst.splice_after(args)</td>
    </tr>
    <tr>
    	<td>(p, lst2)</td>
        <td>
            p 是一个指向 lst 中元素的迭代器，或一个指向 flst 首前位置的迭代器。<br/>
            函数将 lst2 的所有元素移动到 lst 中 p 之前的位置或是 flst 中 p 之后的位置。<br/>
            将元素从 lst2 中删除。lst2 的类型必须与 lst 或 flst 相同，且不能是同一个链表
        </td>
    </tr>
	<tr>
    	<td>(p, lst2, p2)</td>
        <td>
        	p2 是一个指向 lst 中元素的有效的迭代器。<br/>
            将 p2 指向的元素移动到 lst 中，或将 p2 之后的元素移动到 flst 中。<br/>
            lst2 可以是与 lst 或 flst 相同的链表
        </td>
    </tr>
    <tr>
    	<td>(p, lst2, b, e)</td>
        <td>
            b 和 e 必须表示 lst2 中的合法范围。将个定范围中的元素从 lst2 移动到 lst 或 flst。<br/>
            lst2 与 lst（或 flst）可以是相同的链表，但 p 不能指向给定范围中的元素
        </td>
    </tr>
</table>

