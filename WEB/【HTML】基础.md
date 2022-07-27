# HTML基础
## 目录
[toc]

## 1 HTML
HTML 语言用于描述网页。

* HTML 是指超文本标记语言: **H**yper **T**ext **M**arkup **L**anguage
* HTML 不是一种编程语言，而是一种**标记语言**
* 标记语言是一套**标记标签** (markup tag)
* HTML 使用标记标签来**描述**网页
* HTML 文档包含了 HTML **标签**及**文本**内容
* HTML 文档也叫做**web 页面**

### 1.1 示例（Example）

**HTML 开头（Begin）**

* `!Doctype html` - 声明为HTML5文档
* `<meta charset="utf-8" />` - 声明网页编码

**HTML 元素（Element）**

* 以**开始标签**起始，以**结束标签**终止
* **元素内容**是开始标签与结束标签之间的内容
* 大多数元素可拥有**属性**，以键值对形式出现，比如：name="value"。

一个简单示例：

```html
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>我的HTML</title>
</head>
<body>
	<h1>我的第一个标题</h1>
	<p>我的第一个段落</p>
</body>
</html>
```

具体解析见下文。

### 1.2 标题（Heading）
**语法：**

* `<h1>`-`<h6>` - 标题
* `<hr/>` - 水平线

**注释：**

* 浏览器会自动地在标题的前后添加空行。

**注意：**

* 确保HTML标题标签只用于标题，因为搜索引擎使用标题为网页的结构和内容编制索引。

### 1.3 段落（Paragraph）
**语法：**

* `<p>` - 段落
* `<br/>` - 换行

**注释：**

* 浏览器会自动在段落前后添加空行（`</p>`是块级标签）
* 默认情况下，HTML 会自动地在块级元素前后添加一个额外的空行，比如段落、标题元素前后。

### 1.4 文本格式化（Text）
**语法：**格式化文本

* `<b>` - 加粗
* `<strong>` - 加粗
* `<i>` - 斜体
* `<em>` - 斜体
* `<big>` - 放大
* `<small>` - 缩小
* `<sup>` - 上标
* `<sub>` - 下标
* `<ins>` - 插入字（下划线）
* `<del>` - 删除字（中横线）

**语法：**计算机文本

* `<pre>` - 预格式文本
* `<kbd>` - 键盘输入
* `<tt>` - 打印机文本
* `<code>` - 计算机输出
* `<samp>` - 计算机代码
* `<var>` - 计算机变量

**语法：**书信文本

* `<abbr>` - 缩写（在`title`属性内赋予完整文本）
* `<address>` - 地址
* `<bdo>` - 文字方向（`dir`属性决定方向，`"rtl"`从右到左）
* `<blockquote>` - 长引用
* `<q>` - 短引用
* `<cite>` - 引证（作品标题等）
* `<dfn>` - 项目

### 1.5 链接（Link）
**语法：**

```html
<a href="url">链接元素</a>
```

**属性：**

* `href` - 链接地址（url）
	* `href="#"` - 表示一个空链接
	* `href="#id"` - 可以定位到某个 id 的标签处
* `target` - 链接行为，默认在原窗口打开
	* `"_blank"` - 新窗口打开

**注释：**

* 默认情况下，链接将以以下形式出现在浏览器中：
	* 一个未访问过的链接显示为蓝色字体并带有下划线
	* 访问过的链接显示为紫色并带上下划线
	* 点击链接时，链接显示为红色并带上下划线
	* 如果为这些超链接设置了CSS样式，展示样式会根据CSS的设定来显示

**注意：**

* 始终将正斜杠添加到子文件夹。否则服务器会产生两次HTTP请求。

### 1.6 头部（Head）
**语法：**

* `<head>` - 包括所有的头部元素
* `<meta>` - 基本元数据（网页描述="description"，搜索引擎关键词="keywords"，修改时间，作者="author"等）
	* `name` - 数据名称
	* `content` - 数据内容
	* `http-equiv` - 页面动作，如间隔刷新页面`http-equiv="refresh"`，然后`content="30"`指定秒数
* `<title>` - 标题，定义了以下场所的标题
	* 浏览器工具栏的标题
	* 收藏夹的标题
	* 搜索引擎结果页面的标题
* `<base />` - 基本链接地址，作为所有链接标签的默认链接
	* `href` - 链接地址
	* `target` - 链接行为
* `<link>` - 文档与外部资源的关系，通常用于连接到样式表
	* `rel` - 关系
	* `type` - 类型
	* `href` - 地址
* `<style>` - 内部样式
* `<script>` - 脚本文件

### 1.7 样式（CSS）
**语法：**

内联样式

```html
<p style="color:blue;margin-left:20px;">This is a paragraph.</p>
```

内部样式

```html
<style type="text/css">
	h1 {color:red;}
	p {color:blue;}
</style>
```

外部样式

```html
<link rel="stylesheet" type="text/css" href="sytle.css" />
```

**典例：**

A 无下划线的链接

```html
<a href="http://www.puresakura.com" style="text-decoration:none;">纯樱</a>
```

B 背景

```html
<div style="background-color:red;">一个块级元素</div>
```

C 字体

```html
<div style="font-family:verdana;color:red;font-size:20px;">一个块级元素</div>
```

D 文字对齐

```html
<div style="text-align:center">一个块级元素</div>
```

**注意：**

* 元素样式之间以分号结尾
* 某些标签无法通过父级标签来改变

更多内容参见“【CSS】CSS基础” 。

### 1.8 图像（Image）
**语法：**

* `<img src="url" alt="text" />` - 图像
	* `src` - 图像地址
	* `alt` - 可替换文本
	* `width` - 图像宽度
	* `height` - 图像高度

**典例：**

A 图像地图

```html
<img src="planets.gif" width="x_number" height="y_number" alt="planets" usemap="#planetmap" />

<map name="planetmap">
	<area shape="rect" coords="x1,y1,x2,y2" href="url" />
	<area shape="circle" coords="x1,y1,r" href="url />
	<area shape="poly" coords="x1,y1,x2,y2 ... ..." href="url" />
</map>
```

### 1.9 表格（Table）
**语法：**

```html
<table>

	<caption>Title</caption>

	<thead>
		<tr>
			<th>Header A</th>
			<th>Header B</th>
		</tr>
	</thead>

	<tbody>
		<tr>
			<th>Header 1</th>
			<th>Header 2</th>
		</tr>
		<tr>
			<td>row 1, cell 1</td>
			<td>row 1, cell 2</td>
		</tr>
		<tr>
			<td>row 2, cell 1</td>
			<td>row 2, cell 2</td>
		</tr>
	</tbody>

	<tfoot>
		<tr>
			<td>Footer A</td>
			<td>Footer B</td>
		</tr>
	</tfoot>

</table>
```

**属性：**

* `border` - 表格属性，表格边框
* `cellpadding` - 表格属性，单元格边距
* `cellspacing` - 表格属性，单元格间距
* `colspan` - 单元格属性，合并单元格，表示行跨格数
* `rowspan` - 单元格属性，合并单元格，表示列跨格数

### 1.10 列表（List）
**语法：**

无序列表

```html
<ul>
	<li>Coffee</li>
	<li>Milk</li>
</ul>
```

有序列表

```html
<ol>
	<li>Coffee</li>
	<li>Milk</li>
</ol>
```

自定义列表

```html
<dl>
	<dt>Class A</dt>
		<dd>Item 1</dd>
		<dd>Item 2</dd>
	<dt>Class B</dt>
		<dd>Item 1</dd>
		<dd>Item 2</dd>
</dl>
```

### 1.11 区块（Block）
**块级元素（Block-level）：**

块级元素占用一行，通常以新行开始/结束。

**内联元素（Inline）：**

内联元素占用当前元素块，通常不会以新行开始。

**一般区块：**

**语法：**

* `<div>` - 块级元素
* `<span>` - 内联元素

**Tips：**

* 如果与 CSS 一同使用，`<div>` 元素可用于对大的内容块设置样式属性。
* `<div>` 元素的另一个常见的用途是文档布局。由于创建高级的布局非常耗时，使用模板是一个快速的选项。
* 当与 CSS 一同使用时，`<span>` 元素可用于为部分文本设置样式属性。

### 1.12 表单（Form）
**语法：**

* `<form name="名称" action="url" method="get/post">` - 表单，包含表单元素的区域
	* `name` - 表单名称
	* `action` - 表单提交地址，定义了目的文件的文件名，这个文件通常会对接收到的输入数据进行相关的处理
	* `method` - 表单提交方式，`get`/`post`

下面是表单内容：

* 文本域

   ```html
   <input type="text" name="键名" />
   ```

* 密码字段

   ```html
   <input type="password" name="键名" />
   ```

* 单选按钮

   ```html
   <input type="radio" name="键名1" value="预选值" checked="checked" />预选值
   	<input type="radio" name="键名1" value="可选值1" />可选值1
   	<input type="radio" name="键名1" value="可选值2" />可选值2
   ```

* 复选框

   ```html
   <input type="checkbox" name="键名1" value="预选值" checked="checked" />预选值
   	<input type="checkbox" name="键名1" value="可选值1" />可选值1
   	<input type="checkbox" name="键名1" value="可选值2" />可选值2
   ```

* 自定义按钮

   ```html
   <input type="button" value="按钮名" />
   ```

* 文件上传

   ```html
   <input type="file" name="键名" />
   ```

* 提交按钮

   ```html
   <input type="submit" value="Submit" />
   	<input type="image" src="文件路径" alt="替换文字" />
   ```

* 重置按钮

   ```html
   <input type="reset" value="Reset" />
   ```

* 下拉列表

   ```html
   <select name="键名1">
   	<option value="预选值" selected>预选值</option>
   	<option value="可选值1">可选值1</option>
   	<option value="可选值2">可选值2</option>
   	</select>
   ```

* 文本域

		```html
		<textarea row="行数" cols="列数">预置内容</textarea>
	```

**典例：**

A 带边框表单

```html
<form action="" method="">
<fieldset>
<legend>Form Legend</legend>
input_text_1: <input type="text" size="30" /> </br>
input_text_2: <input type="text" size="30" />
</fieldset>
</form>
```

B 表单发送电子邮件

```html
<form action="MAILTO:邮箱地址" method="post" enctype="text/plain">
	Name:<br/>
	<input type="text" name="name" value="名称" /><br/>
	E-mail:<br/>
	<input type="text" name="mail" value="邮箱" /><br/>
	Comment:<br/>
	<input type="text" name="comment" value="内容" /><br/><br/>
	<input type="submit" value="发送" />
	<input type="reset" value="重置" />
</form>
```

### 1.13 框架（Inner Frame）
**语法：**

```html
<iframe src="url" width="宽度" height="高度" frameborder="边框大小"></iframe>
```

### 1.14 颜色（Color）
**语法：**

* `#abcdef` - 颜色十六进制
* `rgb(x,y,z)` - 颜色RGB
* `rgba(x,y,z,t)` - 颜色RGB+透明度(0~1)
* `颜色名` - 17标准颜色+124其他颜色

### 1.15 脚本（Script）
JavaScript 最常用于图片操作、表单验证以及内容动态更新。

**语法：**

内部脚本

```html
<script>
	内部脚本
</script>
```

外部脚本

```html
<script src="url"></script>
```

**注释：**

* 如果使用 "src" 属性，则 `<script>` 元素必须是空的。

更多内容参见 【JavaScript】JavaScript基础

### 1.16 字符实体（Character）
**语法：**

音标

* `&#768;` - 音标符，第四声
* `&#769;` - 音标符，第二声
* `&#770;` - 音标符，尖
* `&#771;` - 音标符，波浪

数学

* `&lt;` - 小于号
* `&gt;` - 大于号
* `&times;` - 乘号
* `&divide;` - 除号
* `&#176;` - 度
* `&#8451;` - 摄氏度

排版

* `&amp;` - 和号
* `&quot;` - 引号
* `&apos;` - 撇号
* `&nbsp;` - 不间断空格

其他

* `&cent;` - 分
* `&pound;` - 镑
* `&yen;` - 人民币/日元
* `&euro;` - 欧元
* `&sect;` - 小节
* `&copy;` - 版权
* `&reg;` - 注册商标
* `&trade;` - 商标

**注释：**

* 实体名称对大小写敏感！

### 1.17 统一资源定位符（URL）
**语法：**

```
scheme://host.domain:port/path/filename
```

* `scheme` - 因特网服务类型（如 http，https，ftp，file）
* `host` - 域主机（如 http默认 www）
* `domain` - 因特网域名（如 puresakura.com）
* `:port` - 主机端口号（如 http默认 80）
* `path` - 服务器上路径（如 根目录 /）
* `filename` - 资源名称

**注意：**

* URL只能使用ASCII字符集
* URL使用`%`跟随两位十六进制数来替换非ASCII字符
* URL使用`+`表示空格

## 1.18 外部文件的路径
考虑以下文件结构：

```
+ web
	+ css
	- html
		- hello.html
	- images
		- example.jpg
```

* 绝对路径 - 相对于Web服务器根目录，以`/`开头

   ```html
   <img src="/web/images/example.jpg" />
   ```

* 相对路径 - 相对于本目录或自定义目录

   ```html
   <!--相对于本文件-->
   	<img src="../images/example.jpg/" />
   <!--指定一个base tag，并且结尾要有斜杠"/"来表示这是一个目录-->
   <!--head部分-->
   <base href="localhost:8000/web/" />
   <!--body部分-->
   <img src="images/example.jpg" />
   ```
## 2 HTML5
HTML5 是下一代 HTML 标准。

HTML5 仍处于完善之中。然而，大部分现代浏览器已经具备了某些 HTML5 支持。

HTML5 受包括Firefox（火狐浏览器），IE9及其更高版本，Chrome（谷歌浏览器），Safari，Opera等国外主流浏览器的支持；国内的傲游浏览器（Maxthon）， 360浏览器、搜狗浏览器、QQ浏览器、猎豹浏览器等同样具备支持HTML5的能力。

### 2.1 图形绘制（Canvas)
什么是 Canvas?

* HTML5 元素用于图形的绘制，通过脚本 (通常是JavaScript)来完成.
* 标签只是图形容器，您必须使用脚本来绘制图形。
* 可以通过多种方法使用Canva绘制路径,盒、圆、字符以及添加图像。

**语法：**

创建画布与环境

```html
<canvas id="身份" width="宽度" height="高度" style="border:1px solid #000000;">不支持时显示的内容</canvas>

<script>
<!--1 获取元素-->
var c=doucment.getElementById("身份");
<!--2 获取环境（内有多种绘制路径、矩形、圆形、字符及添加图像的方法）-->
var ctx=c.getcontext("2d");
</script>
```

* `width` - `canvas`属性，画布宽度
* `height` - `canvas`属性，画布高度
* `getContext(str)` - 获取指定类型的画布
	* `2d` - 2D画布
	* `webgl` - 3D画布

**绘图方法：**

**A 线条**

* `strokeStyle` - 线条填充属性，可以是CSS颜色
* `moveTo(x,y)` - 线条开始坐标
* `lineTo(x,y)` - 线条结束坐标
* `stroke()` - 画线（根据上述两个方法的起止点）

案例演示

```javascript
var c=document.getElementById("身份");
var ctx=c.getContext("2d");
ctx.moveTo(x1,y1);
ctx.lineTo(x2,y2);
ctx.stroke();
```

**B 矩形**

* `fillstyle` - 填充属性，可以是CSS颜色，渐变或图案，默认设置是#000000（黑色）
* `clearRect(x,y,width,heighe)` - 清除矩形内的像素，变透明
* `fillRect(x,y,width,height)` - 矩形
	* `x` - 左上角顶点横坐标
	* `y` - 左上角顶点纵坐标
	* `width` - 矩形宽度
	* `height` - 矩形高度

案例演示

```javascript
var c=document.getElementById("身份");
var ctx=c.getcontext("2d");
<!--设置填充-->
ctx.fillStyle="CSS颜色，渐变或图案";
<!--绘制矩形-->
ctx.fillRect(x,y,width,height);
```

**C 圆形**

* `strokeStyle` - 线条填充属性，可以是CSS颜色
* `beginPath()` - 开始一段路径
* `arc(x,y,r,start,stop,counterclockwise)` - 圆形
	* `x` - 圆心横坐标
	* `y` - 圆心纵坐标
	* `r` - 圆半径
	* `start` - 起始角度（弧度，x轴为0度）
	* `stop` - 结束角度（弧度，x轴为0度）
		* 注 - `Math.PI`表示180&#176;，画圆的方向是顺时针
	* `counterclockwise` - 可选。规定`False=顺时针`，`true`=逆时针
* `stroke()` - 画线（根据arc的参数）

案例演示

```javascript
var c=document.getElementById("身份");
ctx = c.getContext("2d");
ctx.beginPath();
ctx.arc(x,y,r,start,stop);
ctx.stroke();
```

**D 文本**

* `font` - 字体属性
* `shadowOffsetX` - 阴影x轴偏移量
* `shadowOffSetY` - 阴影y轴偏移量
* `shadowBlur` - 阴影模糊量
* `shadowColor` - 阴影颜色
* `fillText(text,x,y)` - 绘制实心文本
* `strokeText(text,x,y)` - 绘制空心文本

案例演示

```javascript
var c=document.getElementById("身份");
var ctx=c.getContext("2d");
ctx.font="30px Arial";
ctx.fillText("Hello World!",10,50);
ctx.strokeText("Hello World!",10,85);
```

**E 渐变**

* `createLinearGradient(x1,y1,x2,y2)` - 线性渐变
	* `x1` - 起点横坐标
	* `y1` - 起点纵坐标
	* `x2` - 终点横坐标
	* `y2` - 终点纵坐标
* `createRadialGradient(x1,y1,r1,x2,y2,r2)` - 径向渐变
	* `x1` - 起点圆心横坐标
	* `y1` - 起点圆心纵坐标
	* `r1` - 起点圆半径
	* `x2` - 终点圆心横坐标
	* `y2` - 终点圆心纵坐标
	* `r2` - 终点圆半径
* `addColorStop(stop,color)` - 规定渐变对象中的颜色和位置，可以使用多次
	* `stop` - 介于 0.0 与 1.0 之间的值，表示渐变中开始与结束之间的位置
	* `color` - 在`stop`位置显示的CSS颜色值

案例演示

```javascript
var c=document.getElementById("身份");
var ctx=c.getContext("2d");

var grd=ctx.createLinearGradient(x1,y1,x2,y2);
grd.addColorStop(0,"black");
grd.addColorStop("0.3","magenta");
grd.addColorStop("0.5","blue");
grd.addColorStop("0.6","green");
grd.addColorStop("0.8","yellow");
grd.addColorStop(1,"red");

ctx.fillStyle=grd;
fillRect(x,y,width,height);
```

**F 图像**

* `drawImage(img,sx,sy,swidth,sheight,x,y,width,height)` - 图像
	* `img` - 图像，画布或视频
	* `sx` - 裁剪起点横坐标，可选
	* `sy` - 裁剪起点纵坐标，可选
	* `swidth` - 裁剪宽度，可选
	* `sheight` - 裁剪高度，可选
	* `x` - 图像左上角顶点横坐标
	* `y` - 图像左上角顶点纵坐标
	* `width` - 图像宽度，可选
	* `height` - 图像高度，可选

案例演示

```javascript
var c=document.getElementById("身份");
var ctx=c.getContext("2d");
var img=document.getElementById("媒体身份");
ctx.drawImage(img,x,y);
```

### 2.2 可伸缩矢量图形（SVG）
什么是SVG？

* SVG 指可伸缩矢量图形 (Scalable Vector Graphics)
* SVG 用于定义用于网络的基于矢量的图形
* SVG 使用 XML 格式定义图形
* SVG 图像在放大或改变尺寸的情况下其图形质量不会有损失
* SVG 是万维网联盟的标准
* SVG 与 DOM 和 XSL 之类的 W3C 标准是一个整体

**语法：**

内联SVG

```html
<svg xmlns="url" version="版本" height="高度">
	<polygon points="x1,y1 x2,y2 x3,y4 ... ..." style="style1:value1;style2:value2;... ...;" />
</svg>
```

详细内容参见“【HTML】SVG”。

### 2.3 数学标记语言（MathML）
**语法：**

* `<math xmlns="ulr">` - 数学，包含MathML元素的区域
* `<mrow>......</mrow>` - 用于包裹一个或多个表达式（可省略）。
* `<msup>......</msup>` - 用于包裹上标的表达式（如：指数函数）。
* `<msub>......</msub>` - 用于包裹下表的表达式。
* `<mi>.........</mi>` - 用于包裹字符。
* `<mn>.........</mn>` = 用于包裹数字。
* `<mo>...........</mo>` - 用于包裹各种运算符号（+，-，<mo></mo>,<mfrac></mfrac>，<,>,(,)等）
* `<msqrt>..........</msqrt>` - 用于开根号。
* `<mfenced open="[" close="]">.........</mfenced>` - 用于包裹矩阵即先定义外围的括号。
* `<mtable>..........</mtable>` - 类似table。
* `<mtr>..........</mtr>` - 代表矩阵的行。
* `<mtd>.........</mtd>` - 代表每行的每一个值。

### 2.4 拖放（Drag & Drop）
**语法：**

**A 元素修饰**

```html
<div id="drop_area" ondrop="drop(event)" ondragover="allowDrop(event)"></div>
<br/>
<img id="drag_element" draggable="true" ondragstart="drag(event)" width="width" height="height" />
```

**B 拖动元素**

* `draggable="true"` - 设置元素可拖放
* `ondragstart="drag(event)"` - 设置拖动数据

		function drag(ev) {
			<!--设置拖动数据，数据类型为Text，目标为可拖动元素的id-->
			ev.dataTransfer.setData("Text", ev.target.id);
		} 

**C 放置元素**

* `ondragover="allowDrop(event)"` - 设置元素可放置

		function allowDrop(ev) {
			<!--阻止对元素的默认处理方式-->
			ev.preventDefault();
		}

* `ondrop="drop(event)"` - 获取拖动数据

		function drop(ev) {
			<!--阻止对元素的默认处理方式-->
			ev.preventDefault();
			<!--获取被拖数据--> <!--该方法将返回在 setData() 方法中设置为相同类型的任何数据。-->
			var data=ev.dataTransfer.getData("Text");
			<!--追加被拖元素到放置元素中-->
			ev.target.appendChild(document.getElementById(data));

### 2.5 地理定位（Geolocation）
**语法：**

* `getCurrentPosition(位置显示方法, 错误处理方法)` - 获取用户地理位置
	* `coords.latitude` - 返回值，十进制纬度
	* `coords.longitude` - 返回值，十进制经度
	* `coords.accuracy` - 返回值，位置精度
	* `coords.altitude` - 返回值，海拔
	* `coords.altitudeAccuracy` - 返回值，海拔精度
	* `coords.heading` - 返回值，方向（正北开始以度计算）
	* `coords.speed` - 返回值，速度
	* `coords.timestamp` - 返回值，响应的时间戳
* `watchPostion(位置显示方法, 错误处理方法)` - 获取用户地理位置，并在用户移动时更新位置

案例演示：简单实例

```javascript
var x=document.getElementById("demo");
function getLocation() {
	if (navigator.geolocation) {
		// 此为核心调用语句
		navigator.geolocation.getCurrentPosition(showPosition, showError);
	}
	else {
		x.innerHTML="该浏览器不支持获取地理位置。";
	}
}

function showPosition(position) {
	x.innerHTML="纬度：" + position.coords.latitude + "<br/>经度" + position.coords.longtitude;
}

function showError(error) {
	switch(error.code) {
		case error.PERMISSION_DENIED:
			x.innerHTML="用户拒绝对获取地理位置的请求。";
			break;
		case error.POSITION.UNANAILABLE:
			x.innerHTML="位置信息不可用。";
			break;
		case error.TIMEOUT:
			x.innerHTML="请求用户地理位置超时。";
			break;
		case error.UNKNOWN_ERROR:
			x.innerHTML="未知错误。";
			break;
	}	
}
```

案例演示：在地图中显示结果

```javascript
function showPosition(position) {
	var latlon=position.coords.latitude+","+position.coords.longitude;

	var img_url="http://maps.gooleapis.com/maps/api/staticmap?center="+latlon+"$zoom=14$size=400x300&sensor=false";
	document.getElemById("mapholder").innerHTML="<img src='"+img_url+"'>";
}
```

案例演示：谷歌地图脚本

```html
<!DOCTYPE html>
<html>
<body>
<p id="demo">点击按钮获取您当前坐标（可能需要比较长的时间获取）：</p>
<button onclick="getLocation()">点我</button>
<div id="mapholder"></div>
<script src="//maps.google.com/maps/api/js?sensor=false"></script>
<script>
var x=document.getElementById("demo");
function getLocation()
  {
  if (navigator.geolocation)
	{
	navigator.geolocation.getCurrentPosition(showPosition,showError);
	}
  else{x.innerHTML="该浏览器不支持获取地理位置。";}
  }

function showPosition(position)
  {
  lat=position.coords.latitude;
  lon=position.coords.longitude;
  latlon=new google.maps.LatLng(lat, lon)
  mapholder=document.getElementById('mapholder')
  mapholder.style.height='250px';
  mapholder.style.width='500px';

  var myOptions={
  center:latlon,zoom:14,
  mapTypeId:google.maps.MapTypeId.ROADMAP,
  mapTypeControl:false,
  navigationControlOptions:{style:google.maps.NavigationControlStyle.SMALL}
  };
  var map=new google.maps.Map(document.getElementById("mapholder"),myOptions);
  var marker=new google.maps.Marker({position:latlon,map:map,title:"You are here!"});
  }

function showError(error)
  {
	 switch(error.code) 
	{
	case error.PERMISSION_DENIED:
	  x.innerHTML="用户拒绝对获取地理位置的请求。"
	  break;
	case error.POSITION_UNAVAILABLE:
	  x.innerHTML="位置信息是不可用的。"
	  break;
	case error.TIMEOUT:
	  x.innerHTML="请求用户地理位置超时。"
	  break;
	case error.UNKNOWN_ERROR:
	  x.innerHTML="未知错误。"
	  break;
	}
  }
</script>
</body>
</html>
```

### 2.6 视频（Video）
**语法：**

```html
<video width="长度" height="宽度" controls>
	<source src="资源位置" type="视频格式" />
	不支持时显示的内容。
</video>
```

* `video/mp4` - 视频格式，MP4
* `video/webm` - 视频格式，WebM
* `video/ogg` - 视频格式，Ogg

`<video>`可以通过JavaScript的DOM进行控制，例如：

```html
<!DOCTYPE html> 
<html> 
<body> 

<div style="text-align:center"> 
  <button onclick="playPause()">播放/暂停</button> 
  <button onclick="makeBig()">放大</button>
  <button onclick="makeSmall()">缩小</button>
  <button onclick="makeNormal()">普通</button>
  <br> 
  <video id="video1" width="420">
	<source src="/statics/demosource/mov_bbb.mp4" type="video/mp4">
	<source src="/statics/demosource/mov_bbb.ogg" type="video/ogg">
	您的浏览器不支持 HTML5 video 标签。
  </video>
</div> 

<script> 
var myVideo=document.getElementById("video1"); 

function playPause()
{ 
if (myVideo.paused) 
  myVideo.play(); 
else 
  myVideo.pause(); 
} 

function makeBig()
{ 
myVideo.width=560; 
} 

function makeSmall()
{ 
myVideo.width=320; 
} 

function makeNormal()
{ 
myVideo.width=420; 
} 
</script> 

</body> 
</html>
```

### 2.7 音频（Audio）
**语法：**

```html
<audio controls>
	<source src="资源位置" type="音频格式" />
	不支持时显示的内容。
</audio>
```

* `audio/mpeg` - 音频格式，MP3
* `audio/ogg` - 音频格式，Ogg
* `audio/Wav` - 音频格式，Wav

### 2.8 新输入（New Input）
HTML5 拥有多个新的表单输入类型。这些新特性提供了更好的输入控制和验证。

**语法：**

* `<input type="color" name="键名" />` - 颜色选取
* `<input type="time" name="键名" />` - 时间选择
* `<input type="date" name="键名" />` - 日期选择
* `<input type="datetime" name="键名" />` - 日期时间选择（UTC时间）
* `<input type="datetime-local" name="键名" />` - 日期时间选择（无时区）
* `<input type="month" name="键名" />` - 月份选择
* `<input type="week" name="键名" />` - 周、年选择
* `<input type="email" name="键名" />` - 邮件输入
* `<input type="number" name="键名 min="最小值" max="最大值" />` - 范围内整数输入
* `<input type="range" name="键名" min="最小值" max="最大值" /> ` - 范围内整数滚动条输入
* `<input type="search" name="键名" />` - 搜索字段输入（没啥用，只是输入框）
* `<input type="tel" name="键名" />` - 电话号码字段输入（没啥用，无支持）
* `<input type="url" name="键名" />` - URL地址输入

**注意：**

* 并不是所有浏览器都支持以上类型，此时会显示为常规文本域。
* 有些在新的Web标准中已被废弃

### 2.9 新表单（New Form）
**语法：**

输入选项列表（`<datalist>`）

```html
<input list="列表名" name="键名" />
<datalist id="列表名">
	<option value="可选值1" />
	<option value="可选值2" />
</datalist>
```

秘钥生成（`<keygen>`）

> `<keygen>` 元素的作用是提供一种验证用户的可靠方法。
> 
> `<keygen>` 标签规定用于表单的密钥对生成器字段。

> 当提交表单时，会生成两个键，一个是私钥，一个公钥。

> 私钥（private key）存储于客户端，公钥（public key）则被发送到服务器。公钥可用于之后验证用户的客户端证书（client certificate）。

```html
<keygen name="键名" />
<input type="submit" />
```

类型输出（需结合DOM）（`<output>`）

```html
<form oninput="x.value=parseInt(a.value)+parseInt(b.value)">0
<input type="range" id="a" value="50" />100
+<input type="number" id="b" value="50" />
=<output name="x" for="a b" />
</form>
```

**注意：**

* 并不是所有浏览器都支持以上类型，此时会显示为常规表单
* 有些在新的Web标准中已被废弃

### 2.10 新表单属性（New Form Attributes）
**语法：**

A `<form>`属性：

* `autocomplete="on/off"` - `<form>`与`<input>`属性，自动完成（记录上次输入内容）
* `novalidate` - 表单提交时不验证域
	* 布尔属性（有就表示启用）

B `<input>`属性

* `autofocus` - 页面加载时域自动获得焦点
	* 布尔属性
* `form="表单名"` - 规定输入域所属的一个或多个表单（如需引用一个以上的表单，请使用空格分隔的列表）
* `formaction="URL"` - 描述表单提交的URL地址
	* 注意：用于`type="submit"`和`type="image"`
* `formenctype="编码格式"` - （覆盖）描述表单提交到服务器的数据编码
	* 注意：只针对form中`method="post"`表单
	* 注意：可用于`type="submit"`和`type="image"`
* `formmethod="get/post"` - （覆盖）定义表单提交方式
	* 注意：可用于`type="submit"`和`type="image"`
* `formnovalidate` - （覆盖）描述表单提交时无需验证域
	* 注意：用于`type="submit"`
* `formtarget` - （覆盖）指定页面提交后的展示方式
	* 注意：用于`type="submit"`和`type="image"`
* `height`&`width` - 规定图像高度&宽度
	* 注意：用于`type="image"`
	* 提示：设置高度与宽度，使得页面加载时能预留空间
* `list="datalist"` - 规定输入域的数据列表选项
* `multiple` - 规定元素可选择多个值
	* 布尔属性
	* 注意：用于email 和 file类型的 `<input>`标签
* `pattern="reg"` - 描述一个正则表达式用于验证元素值，**有用！**
	* 注意：用于字段输入型`<input>`
	* 提示：使用`title`属性提示输入格式
* `placeholder="预置值"` - 预置值，**有用！**
	* 注意：用于字段输入型`<input>`
* `required` - 规定输入域不能为空
	* 布尔属性
* `min` & `max` & `step` - 规定数值输入的范围与合法间隔
	* 注意：用于数值输入型`<input>`，date pickers、number 以及 range

### 2.11 语义元素（New Lang-Element）
!["图片被吞掉了！"](img_sem_elements.gif)

**语法：**

* `<section>` - 定义文档中的节（章节，页眉，页脚等）
* `<article>` - 定义独立的内容
* `<nav>` - 定义导航链接部分
* `<aside>` - 定义页面主区域内容之外的内容（侧边栏等）
* `<header>` - 定义内容的介绍展示区域
* `<footer>` - 描述文档的底部区域（作者，著作权信息，链接使用条款等）
* `<figure>` - 规定独立的流内容（图像，图标，照片，代码等），即使被删也不影响文档流
* `<figurecaption>` - 上述元素子元素，定义标题

使生效语句

```css
header, section, footer, aside, nav, article, figure {
	display: block;
}
```

### 2.12 Web存储

### 2.13 Web SQL

### 2.14 应用程序缓存

### 2.15 Web Worker

### 2.16 SSE

### 2.17 WebSocket

## 3 归纳
### 3.1 插入图像
有两种方式可以插入图像：

* `<image>` 标签
* `<canvas>` 标签

其中第二种具有更高的灵活性。

### 3.2 字体颜色

只改颜色：

```html
<font color='red'>红色</font>
```

改字体、改颜色：

```html
<span style='color:文字颜色;background:背景颜色;font-size:文字大小;font-family:字体;'>文字</span>
```

