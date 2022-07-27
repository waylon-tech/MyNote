## 目录

[toc]

## 1 PAC 语法

代理自动配置（英语：Proxy auto-config，简称 PAC）是一种网页浏览器技术，用于定义浏览器该如何自动选择适当的代理服务器来访问一个网址。

一个 PAC 文件包含一个 JavaScript 形式的函数 `FindProxyForURL(url, host)`，这个函数返回一个包含一个或多个访问规则的字符串，用户代理根据这些规则适用一个特定的代理其或者直接访问。并且，当一个代理服务器无法响应的时候，多个访问规则提供了其他的后备访问方法。

浏览器在访问其他页面以前，会首先访问这个 PAC 文件，文件中的 URL 可能是手工配置的，也可能是是通过网页的网络代理自发现协议（Web Proxy Autodiscovery Protocol）自动配置的。

### 1.1 语法

自定义代理规则的设置语法与 GFWlist 相同，语法规则如下：

- 通配符 `*`
  - 比如 `*.example.com/*`
  - 实际书写时可省略 `*` ， 如`.example.com/` 和 `*.example.com/*` 效果一样
- 正则表达式
  - 以 `\` 开始和结束， 如 `\[\w]+:\/\/example.com\`
- 例外规则 `@@`
  - 如 `@@*.example.com/*` 满足 `@@` 后规则的地址不使用代理
- 匹配地址开始和结尾 `|`
  - 如 `|http://example.com` 、 `example.com|` 分别表示以 `http://example.com` 开始和以 `example.com` 结束的地址
- 标记 `||`
  - 如 `||example.com` 则 `http://example.com` 、`https://example.com` 、 `ftp://example.com` 等地址均满足条件。
- 注释 `!`
  - 如 `!我是注释`
- 分隔符 `^`
  - 表示除了字母、数字或者 `_` `-` `.` `%` 之外的任何字符。
  - 如 `http://example.com^` ，`http://example.com/` 和 `http://example.com:8000/` 均满足条件，而 `http://example.com.ar/` 不满足条件。

