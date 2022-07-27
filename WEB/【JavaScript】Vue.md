##目录
* 1 基础
	* 1.1 导入 Vue.js
	* 1.2 声明式渲染
	* 1.3 条件与循环
	* 1.4 处理用户输入
	* 1.5 组件化应用
* 2 进阶
	* 2.1 Vue实例
		* 2.1.1 创建 Vue 实例
		* 2.1.2 实例属性与实例方法
		* 2.1.3 生命周期钩子
	* 2.2 模板语法
		* 2.2.1 插值
		* 2.2.2 指令
	* 2.3 计算属性和侦听属性
		* 2.3.1 计算属性
		* 2.3.2 侦听属性
* 3 Class与Style的绑定
	* 3.1 绑定 HTML Class
	* 3.2 绑定 HTML Style
* 4 条件渲染
	* 4.1 `v-if`
	* 4.2 `v-show`
* 5 列表渲染
	* 5.1 `v-for`
	* 5.2 更新检测
	* 5.3 `v-for` 的运用
* 6 事件处理
	* 6.1 处理方式
	* 6.2 修饰符
* 7 表单输入绑定
	* 7.1 基础用法
	* 7.2 值绑定
	* 7.3 修饰符
	* 7.4 组件应用
* 8 组件基础
	* 8.1 基本使用
	* 8.2 参数传递
	* 8.3 使用技巧
* 9 过渡 & 动画
* 10 可复用性 & 组合
	* 10.5 过滤器
* 11 工具
* 12 规模化
* 13 内在
* 14 Cookbook
	* 14.1 使用 axios 访问 API

##1 基础
###1.1 导入Vue.js
用`<script></script>`导入就可以了：

	<!-- 开发环境版本，包含了有帮助的命令行警告 -->
	<script src="..js/vue.js"></script>

	<!-- 生产环境版本，优化了尺寸和速度 -->
	<script src="../vue.min.js"></script>

另有通过服务器加载的方式：

	<!-- 开发环境版本，包含了有帮助的命令行警告 -->
	<script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>

	<!-- 生产环境版本，优化了尺寸和速度 -->
	<script src="https://cdn.jsdelivr.net/npm/vue"></script>

###1.2 声明式渲染
Vue.js 的核心是一个允许采用简洁的模板语法来声明式地将数据渲染进 DOM 的系统。

Vue有两种方式来渲染DOM：

**A 文本插值**

在html中使用 `{{ 变量名 }}` 指定变量：

	<div id="app">
		<h2>{{ message }} are in stock.</h2>
	</div>

**B 属性绑定**

在html标签属性中用 `v-bind:属性="变量名"` 指定变量：

	<div id="app">
		<span v-bind:title="message">
			鼠标悬停显示信息。
		</span>
	</div>

**渲染语法**

在script中用如下语法渲染：
		
	<script>
		var app = new Vue({
			el: '#app',
			data: {
				message: 'Hello world!'
			}
		})
	</script>

注释：

* 上面的例子说明 Vue 可以把数据绑定到 DOM **文本或属性**
* Vue 的**响应性**：在控制台改变`app.message`值，会改变html页面的值。（注意不是 `app.data.message`）
* 但是，在浏览器实际操作的时候，发现没有`app`变量，必须在调试模式点开源码后才行。猜测是为停止Vue加载/运行了。

###1.3 条件与循环
**条件指令**

在标签内使用属性关键字`v-if="判断量"`为该标签添加条件属性，决定标签是否插入/移除

例如：

	<div id="app">
		<h2 v-if="seen">Hello！</h2>
	</div>

	var app3 = new Vue({
	  el: '#app',
	  data: {
	    seen: true
	  }
	})

**循环指令**

在标签内使用关键字`v-for="item in items"`来循环当前标签。

例如：

	<div id="app">
		<ol>
			<li v-for="todo in todos">
				{{ todo.text }}
			</li>
		</ol>
	</div>
	
	var app = new Vue({
			el: '#app',
			data: {
				todos: [
				  { text: '学习 JavaScript' },
				  { text: '学习 Vue' },
				  { text: '整个项目' }
				]
			}
		})

注释：

* 上面的例子说明 Vue 还可以把数据绑定到 DOM **结构**。
* Vue 的**响应性**：在通过 Vue 插入/更新/移除/元素时会自动应用过渡效果。

###1.4 处理用户输入
**(1) `v-on` 事件注册指令**

用于注册事件监听器，用`v-on`绑定监听的属性，该属性值（即变量名）在Vue()的`methods`内处理。

注册语法：

		<element v-on:event="function_name"></element>

响应语法及整体示例：

	<div id="app">
	  <p>{{ message }}</p>
	  <button v-on:click="reverseMessage">逆转消息</button>
	</div>

	var app = new Vue({
	  el: '#app',
	  data: {
	    message: 'Hello Vue.js!'
	  },
	  methods: {
	    reverseMessage: function () {
	      this.message = this.message.split('').reverse().join('')
	    }
	  }
	})

**注释：**

* 响应函数没有触碰 DOM，所有的 DOM 操作都由 Vue 来处理

**(2) `v-model` 表单绑定指令**

表单同步属性，为表单类元素添加`v-model="变量名"`属性，将表单输入与JavaScript变量值同步。

例如：

	<div id="app">
	  <input v-model="message">
	</div>
	
	var app6 = new Vue({
	  el: '#app',
	  data: {
	    message: 'Hello Vue!'
	  }
	})

**注释：**

* Vue 会**自动识别**表单输入元素，然后绑定其内的输入

###1.5 组件化应用
组件系统是 Vue 的另一个重要概念，因为它是一种抽象，允许我们使用小型、独立和通常可复用的组件构建大型应用。

在Vue里，一个组件本质上是一个拥有预定义选项的 Vue 实例。类似于 HTML 的自定义标签/元素。

**组件注册**

	Vue.component('todo-item', {
	  props: ['todo'],
	  template: '<li>{{ todo.text }}</li>'
	})

* `prop` - 组件要用到的属性列表，list
* `template` - 组件渲染的 HTML 模板，str

`todo-item` 组件现在接受一个名为 `todo` 的属性。

**组件使用**

用Vue渲染出响应式环境，然后在里面使用组件，用`v-bind`将`props`的某个属性绑定到Vue环境的某个变量，记得指定主键。

例如：

    <!--
      现在我们为每个 todo-item 提供 todo 对象。todo 对象是变量，即其内容可以是动态的。
      我们也需要为每个组件提供一个“key”，稍后再作详细解释。
    -->	
	<div id="app">
	  <ol>
	    <todo-item
	      v-for="item in groceryList"
	      v-bind:todo="item"	<!-- 此处是核心 -->
	      v-bind:key="item.id">
	    </todo-item>
	  </ol>
	</div>
	
	var app = new Vue({
	  el: '#app',
	  data: {
	    groceryList: [
	      { id: 0, text: '蔬菜' },
	      { id: 1, text: '奶酪' },
	      { id: 2, text: '随便其它什么人吃的东西' }
	    ]
	  }
	})

**注释：**

* 组件有自己独特的作用域，为了使用数据，要把数据通过`props`传递到组件里。
* 属性组 `props` 充当一个接口的角色，将父单元与子单元实现解耦。

本节简单介绍了 Vue 核心最基本的功能，下面的章节是这些功能以及其它高级功能更详细的细节。

##2 进阶
###2.1 Vue 实例
####2.1.1 创建 Vue 实例
在第 1 节已经给出了实例，这里进行详细的叙述。

**语法：**

	var vm = new Vue({
	  // 选项
	})

当创建一个 Vue 实例时，你可以传入一个选项对象。这篇教程主要描述的就是**如何使用这些选项来创建你想要的行为**。

Vue 组件都是 Vue 实例，并且接受相同的选项对象 (一些根实例特有的选项除外)。

####2.1.2 数据与方法
**数据的响应特性**

一个 Vue 实例创建后，其 `data` 对象拥有的属性会加入 Vue 的**响应式系统**（即 data 数据具备响应性）。

响应性是指，当 `data` 对象的属性值改变时，视图（即 HTML）会产生响应匹配更新的值，**反之亦然**。

使用 HTML 的方法 `Object.freeze(变量名)` 会阻止修改现有的属性，即**失去响应性**。

**初始值初始化**

	data: {
	  newTodoText: '',
	  visitCount: 0,
	  hideCompletedTodos: false,
	  todos: [],
	  error: null
	}

注意：使用了`Object.freeze()`后无法再追踪变化。

**自带属性与方法**

Vue自带的实例属性与实例方法其实带有前缀`$`，用于将用户定义的属性区分开来。例如：

	var data = { a: 1 }
	var vm = new Vue({
	  el: '#example',
	  data: data
	})
	
	vm.$data === data // => true
	vm.$el === document.getElementById('example') // => true
	
	// $watch 是一个实例方法
	vm.$watch('a', function (newValue, oldValue) {
	  // 这个回调将在 `vm.a` 改变后调用
	})

**完整的实例属性和方法的列表参考API帮助文档。**

####2.1.3 生命周期钩子
生命周期钩子指出Vue实例创建的各个阶段，可以在各个阶段加入要执行的方法。

**A 例如**：

* **created** - 用来在一个实例被创建之后执行的代码。

	new Vue({
	  data: {
	    a: 1
	  },
	  created: function () {
	    // `this` 指向 vm 实例
	    console.log('a is: ' + this.a)
	  }
	})
	// => "a is: 1"

**B 其他**：

还有其他的生命周期钩子，具体使用参考API帮助文档。

**注意**：

* 不要在选项属性或回调上使用箭头函数，因为箭头函数是和父级上下文绑定在一起的，`this` 的指向不同。
	* 例如：`created: () => console.log(this.a)`
	* 例如：`vm.$watch('a', newValue => this.myMethod())`
	* 导致：`Uncaught TypeError: Cannot read property of undefined`
	* 导致：`Uncaught TypeError: this.myMethod is not a function`

**C 生命周期图示**：

![图片被吞掉了!](vue_lifecycle.png)

###2.2 模板语法
Vue.js 使用了基于 HTML 的模板语法，允许开发者声明式地将 DOM 绑定至底层 Vue 实例的数据。

所有 Vue.js 的模板都是合法的 HTML ，所以能被遵循规范的浏览器和 HTML 解析器解析。

####2.2.1 插值
**(1) 普通文本**：

“Mustache”语法，即双大括号 `{{ 变量名 }}` 修饰的就是文本插值。

使用 `v-once` 属性可执行一次性插值，失去响应性。

例如：

	<span>Message: {{ msg }}</span>
	<span v-once>这个将不会改变: {{ msg }}</span>

**(2) 原始HTML**：

在标签内添加属性 `v-html="变量名"` 会将该标签的内容替换为指定的html内容。

例如：

	<p>Using v-html directive: <span v-html="rawHtml"></span></p>

**注意**：

在站点上动态渲染任何HTML会很危险，因为容易导致**XSS攻击**。所以只对可信内容使用HTML插值，且**绝不要**对用户提供的内容使用插值。

**(3) HTML 属性**

“Mustache”语法不能作用在 HTML 属性上，此时可以使用 `v-bind:属性="变量名"` 指令绑定 HTML 属性值为某个变量。

例如：

	<!-- HTML 的 id 属性值从 dynamicId 中获得 -->
	<div v-bind:id="dynamicId"></div>

对于**布尔属性**的情况，即属性的存在即暗示为 `true`，`v-bind` 指令的工作方式是决定是否渲染到 HTML 元素中。

例如：

	<!--
		如果 isButtonDisabled 的值是 null、undefined 或 false，
		则 disabled 特性甚至不会被包含在渲染出来的 <button> 元素中
	-->
	<button v-bind:disabled="isButtonDisabled">Button</button>

**(4) JavaScript 表达式**
实际上，对于所有的数据绑定，Vue.js 都提供了完全的 JavaScript 表达式支持。

在Vue实例数据的作用域下，用双大括号`{{ 表达式 }}`修饰的表达式、用 `v-bind:属性="表达式"` 绑定的表达式，会作为JavaSrcipt解析，最后会按照文本输出。

注意每个绑定都只能包含单个表达式，否则不会生效。

例如：

	{{ number + 1 }}
	
	{{ ok ? 'YES' : 'NO' }}
	
	{{ message.split('').reverse().join('') }}
	
	<div v-bind:id="'list-' + id"></div>

	<!-- 这是语句，不是表达式 -->
	{{ var a = 1 }}
	
	<!-- 流控制也不会生效，请使用三元表达式 -->
	{{ if (ok) { return message } }}

**注意**：

* 模板表达式都被放在沙盒中，只能访问全局变量的一个白名单，不应该访问用户自定义的全局变量。

####2.2.2 指令
指令 (Directives) 是带有 v- 前缀的特殊 HTML 属性。

指令属性的值预期是单个 JavaScript 表达式 (`v-for` 是例外情况，稍后我们再讨论)。

指令的职责是，当表达式的值改变时，将其产生的连带影响，响应式地作用于 DOM。

**A 参数**：

一个指令接收一个参数，在指令名称之后用冒号表示。

例如：

	<!-- 这里 href 是参数，告知 v-bind 指令将该元素的 href 特性与表达式 url 的值绑定 -->
	<a v-bind:href="url">...</a>

	<a v-on:click="doSomething">...</a>

**B 修饰符**：

修饰符是以半角句号`.`指明的特殊后缀，用于指出一个指令应该以特殊方式绑定。

例如：

	<!-- .prevent 修饰符告诉 v-on 指令对于触发的事件调用 event.preventDefault() -->
	<form v-on:submit.prevent="onSubmit">...</form>

**C 缩写**

Vue.js 为`v-bind` 和 `v-on` 这两个最常用的指令提供了特定的简写。

`v-bind`缩写：

	<!-- 完整语法 -->
	<a v-bind:href="url">...</a>
	
	<!-- 缩写 -->
	<a :href="url">...</a>

`v-on`缩写：

	<!-- 完整语法 -->
	<a v-on:click="doSomething">...</a>
	
	<!-- 缩写 -->
	<a @click="doSomething">...</a>

###2.3 计算属性和侦听属性
####2.3.1 计算属性
计算属性用于**简化表达式**的复杂逻辑。

**A 计算属性 getter**：

在Vue的 `computed` 选项内，为要计算的**新属性**指定 getter 函数。

例如：

	<div id="example">
	  <p>Original message: "{{ message }}"</p>
	  <p>Computed reversed message: "{{ reversedMessage }}"</p>
	</div>
	
	var vm = new Vue({
	  el: '#example',
	  data: {
	    message: 'Hello'
	  },
	  computed: {
	    // 计算属性的 getter
	    reversedMessage: function () {
	      // `this` 指向 vm 实例
	      return this.message.split('').reverse().join('')
	    }
	  }
	})

**B 计算属性 setter**

计算属性默认只有 getter ，不过在需要时你也可以提供一个 setter 函数。

例如：

	// ...
	computed: {
	  fullName: {
	    // getter
	    get: function () {
	      return this.firstName + ' ' + this.lastName
	    },
	    // setter
	    set: function (newValue) {
	      var names = newValue.split(' ')
	      this.firstName = names[0]
	      this.lastName = names[names.length - 1]
	    }
	  }
	}
	// ...

**C 一些区别**

计算属性`computed`与方法属性`methods`可以实现相同的效果：

	<p>Reversed message: "{{ reversedMessage() }}"</p>

	// 在组件中
	methods: {
	  reversedMessage: function () {
	    return this.message.split('').reverse().join('')
	  }
	}

但是两者在底层缓存上有区别：

* `computed`是基于他们的依赖进行缓存的，只在相关依赖发生改变时它们才会重新求值。
	* 因此，在 `message` 值未改变时，多次访问 `reversedMessage` 计算属性不会重新计算。
* `methods`在每次触发重新渲染时，每次调用方法会再次执行函数。

####2.3.2 侦听器
侦听属性用于监听一个变量的变化。

**A 侦听属性**

在Vue的 `watch` 选项内，为要侦听的**属性**指定 动作 函数。

	<div id="demo">{{ fullName }}</div>
	
	var vm = new Vue({
	  el: '#demo',
	  data: {
	    firstName: 'Foo',
	    lastName: 'Bar',
	    fullName: 'Foo Bar'
	  },
	  watch: {
	    firstName: function (val) {
	      this.fullName = val + ' ' + this.lastName
	    },
	    lastName: function (val) {
	      this.fullName = this.firstName + ' ' + val
	    }
	  }
	})

**B 一些区别**：

计算属性`computed`与侦听属性`watch`都可以实现数据间的依赖关系，但是两者在特定代码逻辑上有区别：

* `computed`是将函数`return`返回值直接赋予新的依赖变量。

* `wathc`是侦听一个变量，依赖变量要命令式更新，且变量增加的时候有可能造成代码重复。

计算属性适用于大多数情况，但在需要自定义侦听器，执行**异步**或**开销较大**的操作时，使用侦听属性（侦听器）最有用。

例如：

	<div id="watch-example">
	  <p>
	    Ask a yes/no question:
	    <input v-model="question">
	  </p>
	  <p>{{ answer }}</p>
	</div>
	
	<!-- 因为 AJAX 库和通用工具的生态已经相当丰富，Vue 核心代码没有重复 -->
	<!-- 提供这些功能以保持精简。这也可以让你自由选择自己更熟悉的工具。 -->
	<script src="https://cdn.jsdelivr.net/npm/axios@0.12.0/dist/axios.min.js"></script>
	<script src="https://cdn.jsdelivr.net/npm/lodash@4.13.1/lodash.min.js"></script>
	<script>
	var watchExampleVM = new Vue({
	  el: '#watch-example',
	  data: {
	    question: '',
	    answer: 'I cannot give you an answer until you ask a question!'
	  },
	  watch: {
	    // 如果 `question` 发生改变，这个函数就会运行
	    question: function (newQuestion, oldQuestion) {
	      this.answer = 'Waiting for you to stop typing...'
	      this.debouncedGetAnswer()
	    }
	  },
	  created: function () {
	    // `_.debounce` 是一个通过 Lodash 限制操作频率的函数。
	    // 在这个例子中，我们希望限制访问 yesno.wtf/api 的频率
	    // AJAX 请求直到用户输入完毕才会发出。想要了解更多关于
	    // `_.debounce` 函数 (及其近亲 `_.throttle`) 的知识，
	    // 请参考：https://lodash.com/docs#debounce
	    this.debouncedGetAnswer = _.debounce(this.getAnswer, 500)
	  },
	  methods: {
	    getAnswer: function () {
	      if (this.question.indexOf('?') === -1) {
	        this.answer = 'Questions usually contain a question mark. ;-)'
	        return
	      }
	      this.answer = 'Thinking...'
	      var vm = this
	      axios.get('https://yesno.wtf/api')
	        .then(function (response) {
	          vm.answer = _.capitalize(response.data.answer)
	        })
	        .catch(function (error) {
	          vm.answer = 'Error! Could not reach the API. ' + error
	        })
	    }
	  }
	})
	</script>

在这个示例中，使用 watch 选项允许我们执行异步操作 (访问一个 API)，限制我们执行该操作的频率，并在我们得到最终结果前，设置中间状态。这些都是计算属性无法做到的。

##3 Class 与 Style 的绑定
第 2.1 节模板中，叙述了通过 `v-bind` 绑定 HTML 的属性，从而也可以绑定 class 与 style：只需要通过表达式计算出字符串结果即可。

不过，字符串拼接麻烦且易错。因此，在将 `v-bind` 用于 class 和 style 时，Vue.js 做了专门的增强：

	v-bind:class

	v-bind:style

表达式结果的类型除了字符串之外，还可以是对象或数组。

###3.1 绑定HTML Class
**(1) 绑定到对象**

传给 `v-bind:class` 一个对象，以动态地切换 class。

模板：

	<!-- 可以与普通的 class 属性共存 -->
	<div
		class="static"
		v-bind:class="{ active: isActive, 'text-danger': hasError }"
	</div>

数据：

	// 通过数据的 `true` 或 `false` 以决定是否加入该类
	data: {
		isActive: true,
		hasError: false
	}

会渲染为：

	<div class="static active"></div>

**技巧：**绑定的对象不必内联定义在模板里

* 使用 `data` 选项存放对象

		<div v-bind:class="classObject"></div>
		
		data: {
		  classObject: {
		    active: true,
		    'text-danger': false
		  }
		}

* 使用 `computed` 选项计算对象

		<div v-bind:class="classObject"></div>
	
		data: {
			isActive: true,
			error: null
			}
		},
		computed: {
			classObject: function() {
				return {
					active: this.isActive && !this.error,
					'text-danger': this.error && this.error.type === 'fatal'
				}
			}
		}

**(2) 绑定到数组**

把一个数组传给 v-bind:class，以应用一个 class 列表。

模板：

	<div b-bind:class="[activeClass, errorClass]"><div>

数据：

	data: {
		activeClass: 'active',
		errorClass: 'text-danger'
	}

渲染为：

	<div class="active text-danger"></div>/

**技巧：**

* 使用三元运算符

		<div v-bind:class="[isActive ? activeClass : '', errorClass"></div>

* 在数组语法中使用对象语法

		<div v-bind:class="[{ active: isActive }, errorClass]"></div>

**(3) 组件上的应用**

当在一个自定义组件上使用 class 属性时，这些类将被添加到该组件的根元素上面。这个元素上已经存在的类不会被覆盖。

例如：

声明组件：

	Vue.component('my-component', {
	  template: '<p class="foo bar">Hi</p>'
	})

然后添加一些 class：

	<!-- 普通 class 属性 -->
	<my-component class="baz boo"></my-component>
	<!-- 数据绑定 class 属性 -->
	<my-component v-bind:class="{ active: isActive }"></my-component>

渲染为：

	<!-- 与普通 class 属性共存 -->
	<p class="foo bar baz boo">Hi</p>
	<!-- 与数据绑定 class 属性共存 -->
	<p class="foo bar active">Hi</p>

###3.2 绑定 HTML Style
**(1) 绑定到对象**：

传入一个 JavaScript 对象，直观上非常像 CSS：

	<div v-bind:style="{ color: activeColor, fontSize: fontSize + 'px' }></div>

	data: {
		activeColor: 'red',
		fontSize: 30
	}

**技巧：**绑定的对象不必内联定义在模板里

* 使用 `data` 选项存放对象

		<div v-bind:style="styleObject"></div>
	
		data: {
			styleObject: {
				color: 'red',
				fontSize: '13px',
				'background-color': "rgba(27, 31, 34, 0.85)"
			}
		}

* 使用 `computed` 选项计算对象（略）

**注释：**

* CSS 属性名可以用驼峰式 (camelCase) 或短横线分隔 (kebab-case，记得用单引号括起来)

**(2) 绑定到数组**：

通过数组将多个样式对象应用到同一个元素上：

	<div v-bind:style="[baseStyles, overridingStyles]"></div>

	data: {
		baseStyles: {
			color: 'red',
			fontSize: '13px'
		}
	}

**(3) 自动添加前缀与多重值**

当 v-bind:style 使用需要**添加浏览器引擎前缀的 CSS 属性**时，如 transform，Vue.js 会**自动侦测并添加**相应的前缀。

可以为 style 绑定中的属性**提供一个包含多个值的数组**，常用于提供多个带前缀的值，例如：

	<div :style="{ display: ['-webkit-box', '-ms-flexbox', 'flex'] }"></div>

这样写只会渲染数组中**最后一个**被浏览器支持的值。

##4 条件渲染
###4.1 `v-if`
条件渲染使用条件指令`v-if`以及配套的`v-else`，`v-else-if`。

	<div v-if="type === 'A'">
	  A
	</div>
	<div v-else-if="type === 'B'">
	  B
	</div>
	<div v-else-if="type === 'C'">
	  C
	</div>
	<div v-else>
	  Not A/B/C
	</div>

**技巧：**切换多个元素

* 添加 `v-if` 条件到 `<template>` 当做不可见的包裹元素

		<template v-if="ok">
		  <h1>Title</h1>
		  <p>Paragraph 1</p>
		  <p>Paragraph 2</p>
		</template>

最终渲染将不包含`<template>`元素。

**特性：元素复用**

Vue 会尽可能高效地渲染元素，通常会复用已有元素而不是从头开始渲染。

这么做除了使 Vue 变得非常快之外，还有其它一些好处。例如，如果你允许用户在不同的登录方式之间切换，那么其中不会清除用户已经输入的内容。

若要求重新渲染，只需给不要复用的标签添加一个具有唯一值的 key 属性即可:

		<input placehodler="Enter your email address" key="email-input">

###4.2 `v-show`
用法与`v-if`类似，例如

	<h1 v-show="ok">Hello!</h1>

不同的是`v-show`元素使用会被渲染并保留在DOM中，因为`v-show`只是简单切换 CSS 属性`display`。

且`v-if`具有更高的切换开销，`v-show`具有更高的初始渲染开销。

**注意：**

* `v-show` 不支持 `<template>` 元素，也不支持 `v-else`。
* 不推荐同时使用 `v-if` 和 `v-for`。请查阅风格指南以获取更多信息。

##5 列表渲染
###5.1 `v-for`
`v-for` 作用是把一个 JavaScript 数组/对象对应为一组 HTML 元素。

**(1) 迭代数组**

在`v-for`块中，拥有对父作用域属性的完全访问权限。`v-for`还支持可选的第二个参数`index`作为索引。

模板：

	<ul id="example">
	  <li v-for="(item, index) in items">
	    {{ parentMessage }} - {{ index }} - {{ item.message }}
	  </li>
	</ul>

数据：

	var example = new Vue({
	  el: '#example',
	  data: {
	    parentMessage: 'Parent',
	    items: [
	      { message: 'Foo' },
	      { message: 'Bar' }
	    ]
	  }
	})

**(2) 迭代对象**

也可以用 `v-for` 通过一个对象的属性来迭代。`v-for`还支持可选的第二个参数`key`作为键名，第三个参数`index`为索引。

模板：

	<ul id="v-for-object" class="demo">
	  <li v-for="(value, key, index) in object">
	    {{ index }}. {{ key }}: {{ value }}
	  </li>
	</ul>

数据：

	new Vue({
	  el: '#v-for-object',
	  data: {
	    object: {
	      firstName: 'John',
	      lastName: 'Doe',
	      age: 30
	    }
	  }
	})

**注意：**

* 在遍历对象时，是按 `Object.keys()` 的结果遍历，但是不能保证它的结果在不同的 JavaScript 引擎下是一致的。

**技巧：**使用`key`属性跟踪每个项的身份

`v-for`正在更新以渲染过的元素列表时，如果数据顺序被改变，Vue不会跟踪，而是“就地复用”。

使用属性`key`可以跟踪各项身份，需要用 v-bind 来动态绑定(在这里使用简写)唯一的 id：

		<div v-for="item in items" v-bind:key="item.id">
			<!--内容-->
		</div>

建议尽可能提供`key`，除非遍历输出的 DOM 内容非常简单，或者是刻意依赖默认行为以获取性能上的提升。

###5.2 更新检测
####5.2.1 数组变异方法
Vue 包含一组观察数组的变异方法，调用时会改变所调用的数组。诸如：

* `push()`
* `pop()`
* `shift()`
* `unshift()`
* `splice()`
* `sort()`
* `reverse()`

非变异方法，反之，总是返回一个新数组。可以采用替换操作改变原数组，且该操作非常高效。

* `filter()`
* `concat()`
* `slice()`

####5.2.2 更新检测注意事项
**A 数组更新检测**：

由于JavaScript的限制，Vue不能检测以下**数组变动**：

* 利用索引值直接设置一个项：

		vm.items[indexOfItem] = newValue

* 修改数组长度：

		vm.itmes.length = newLength

为此，可以使用以下方法

* **`Vue.set(vm.items, indexOfItem, newValue)`** - 响应式更新数组，与下者等价
* **`vm.$set(vm.items, indexOfItem, newValue)`** - 响应式更新数组，与上者等价
* **`vm.items.splice(newLength)`** - 响应式改变数组长度

**B 对象更新检测**：

由于JavaScript的限制，Vue不能检测**对象属性**的添加或删除。

为此，可以使用以下方法：

* **`Vue.set(object, key, value)`** - 向嵌套对象响应式添加属性，与下者等价
* **`Vm.$set(object, key, value)`** - 向嵌套对象响应式添加属性，与上者等价
* **`Object.assign()`** - 为对象响应式添加多个属性，**注意使用方式**：

		var vm = new Vue({
			data: {
				userProfile: {
					name: 'Anika'
				}
			}
		})

		vm.userProfile = Onject.assign({}, vm.userProfile, {
			age: 27,
			favoriterColor: 'Vue Green'
		})

###5.3 `v-for`的运用
**A 过滤与排序**：

为获得一个数组的过滤或排序副本，而不实际改变或重置原始数据，可以通过`computed`或`methods`属性创建返回过滤或排序的数组。

例如：

	<li v-for="n in evenNumbers">{{ n }}</li>
	
	data: {
	  numbers: [ 1, 2, 3, 4, 5 ]
	},
	computed: {
	  evenNumbers: function () {
	    return this.numbers.filter(function (number) {
	      return number % 2 === 0
	    })
	  }
	}

在计算属性不适用的情况下(例如，在嵌套 `v-for` 循环中)，可以使用一个 `method` 方法：

	<li v-for="n in even(numbers)">{{ n }}</li>
	
	data: {
	  numbers: [ 1, 2, 3, 4, 5 ]
	},
	methods: {
	  even: function (numbers) {
	    return numbers.filter(function (number) {
	      return number % 2 === 0
	    })
	  }
	}

**B 整数取值**：

如下：

		<div>
		  <span v-for="n in 10">{{ n }} </span>
		</div>

**C 整组渲染**：

类似于 `v-if`，你也可以利用带有 `v-for` 的 `<template>` 渲染多个元素。

如下：

		<ul>
		  <template v-for="item in items">
		    <li>{{ item.msg }}</li>
		    <li class="divider" role="presentation"></li>
		  </template>
		</ul>

**D `v-for` 和 `v-if`**

当它们处于同一节点，v-for 的优先级比 v-if 更高，这意味着 v-if 将分别重复运行于每个 v-for 循环中。

当你想为仅有的一些项渲染节点时，这种优先级的机制会十分有用，如下：

	<li v-for="todo in todos" v-if="!todo.isComplete">
	  {{ todo }}
	</li>

上面的代码只传递了未完成的 todos。

而如果你的目的是有条件地跳过循环的执行，那么可以将 v-if 置于外层元素 (或 <template>)上。如：

	<ul v-if="todos.length">
	  <li v-for="todo in todos">
	    {{ todo }}
	  </li>
	</ul>
	<p v-else>No todos left!</p>

**E 组件应用**：

可以像普通元素一样在组件中使用`v-for`，但**必须**要绑定`id`，且数据**必须**通过`props`传送。

因为任何数据都不会被自动传递到组件里，组件有自己独立的作用域。

例如：

	<my-component
	  v-for="(item, index) in items"
	  v-bind:item="item"
	  v-bind:index="index"
	  v-bind:key="item.id"
	></my-component>

在`<ul>`的`<li>`中使用时，只有`<li>`元素会被视为`<ul>`的有效内容，故添加`is="todo-item"`实现自定义组件与`<li>`的重叠，可以避开一些潜在的浏览器解析错误。

详见9.3 解析DOM注意事项

##6 事件处理
在第 1.4 节已经叙述，可以用 `v-on` 指令注册并监听 DOM 事件，并在触发时运行一些 JavaScript 代码。

###6.1 处理方式
**A 直接处理**：

`v-on` 可以接收一个需要调用的方法名称。

	<div id="example-1">
	  <!-- `greet` 是在下面定义的方法名 -->
	  <button v-on:click="greet">Greet</button>
	</div>
	
	var example1 = new Vue({
	  el: '#example-1',
	  data: {
	    name: 'Vue.js'
	  },
	  // 在 `methods` 对象中定义方法
	  methods: {
	    greet: function (event) {
	      // `this` 在方法里指向当前 Vue 实例
	      alert('Hello ' + this.name + '!')
	      // `event` 是原生 DOM 事件
	      if (event) {
	        alert(event.target.tagName)
	      }
	    }
	  }
	})
	
	// 也可以用 JavaScript 直接调用方法
	example2.greet() // => 'Hello Vue.js!'

**B 内联处理**：

可以在内联 JavaScript 语句中调用方法。

	<div id="example-2">
	  <button v-on:click="say('hi')">Say hi</button>
	  <button v-on:click="say('what')">Say what</button>
	</div>
	
	new Vue({
	  el: '#example-2',
	  methods: {
	    say: function (message) {
	      alert(message)
	    }
	  }
	})

有时也需要在内联语句处理器中访问原始的 DOM 事件。可以用特殊变量 $event 把它传入方法。

	<button v-on:click="warn('Form cannot be submitted yet.', $event)">
	  Submit
	</button>
	
	// ...
	methods: {
	  warn: function (message, event) {
	    // 现在我们可以访问原生事件对象
	    if (event) event.preventDefault()
	    alert(message)
	  }
	}

**为了在方法中访问原始的DOM事件，可以用特殊变量`$event`把它传入方法。**

###6.2 修饰符
**A 事件修饰符**：

Vue.js为`v-on`提供了事件修饰符：

* `.stop`

		<!-- 阻止单击事件继续传播 -->
		<a v-on:click.stop="doThis"></a>

* `.prevent`

		<!-- 提交事件不再重载页面 -->
		<form v-on:submit.prevent="onSubmit"></form>
		
		<!-- 修饰符可以串联 -->
		<a v-on:click.stop.prevent="doThat"></a>
		
		<!-- 只有修饰符 -->
		<form v-on:submit.prevent></form>

* `.capture`

		<!-- 添加事件监听器时使用事件捕获模式 -->
		<!-- 即元素自身触发的事件先在此处理，然后才交由内部元素进行处理 -->
		<div v-on:click.capture="doThis">...</div>

* `.self`

		<!-- 只当在 event.target 是当前元素自身时触发处理函数 -->
		<!-- 即事件不是从内部元素触发的 -->
		<div v-on:click.self="doThat">...</div>

* `.once`

		<!-- 点击事件将只会触发一次 -->
		<a v-on:click.once="doThis"></a>

* `.passive` （尤其能够提升移动端的性能）

		<!-- 滚动事件的默认行为 (即滚动行为) 将会立即触发 -->
		<!-- 而不会等待 `onScroll` 完成  -->
		<!-- 这其中包含 `event.preventDefault()` 的情况 -->
		<div v-on:scroll.passive="onScroll">...</div>

注意：

修饰符的顺序很重要，相应的代码会以同样的顺序产生。

`.prevent`与`.passive`不能一起使用，因为 `.prevent` 将会被忽略，同时浏览器可能会向你展示一个警告。

**B 按键修饰符**：

常规键：配合`v-on:keyup`捕获按键输出

* `.enter`
* `.tab`
* `.delete` - 捕获“删除”和“退格”键
* `.esc`
* `.space`
* `.up`
* `.down`
* `.left`
* `.right`

系统键：`v-on:keyup`或`v-on:click`捕获按键输出

* `.ctrl`
* `.alt`
* `.shift`
* `.meta`
* `.exact` - 允许控制由精确的**系统修饰符**组合触发的事件

		<!-- 即使 Alt 或 Shift 被一同按下时也会触发 -->
		<button @click.ctrl="onClick">A</button>
		
		<!-- 有且只有 Ctrl 被按下的时候才触发 -->
		<button @click.ctrl.exact="onCtrlClick">A</button>
		
		<!-- 没有任何系统修饰符被按下的时候才触发 -->
		<button @click.exact="onClick">A</button>

**注意**：`v-on:keyup`只有在按住系统键，释放其他键时才能触发。

**C 鼠标修饰符**：

* `.left`
* `.right`
* `.middle`

##7 表单输入绑定
###7.1 基础用法
可以用 `v-model` 指令在表单 `<input>`、`<textarea>` 及 `<select>` 元素上创建双向数据绑定。

它会根据控件类型自动选取正确的方法来更新元素。

* `input` - 文本 - 绑定静态字符串

		<input v-model="message" placeholder="edit me">
		<p>Message is: {{ message }}</p>

* `textarea` - 多行文本 - 绑定静态字符串（若插值并不会生效，应用 `v-model` 来代替。）

		<span>Multiline message is:</span>
		<p style="white-space: pre-line;">{{ message }}</p>
		<br>
		<textarea v-model="message" placeholder="add multiple lines"></textarea>

* `input radio` - 单选按钮 - 绑定静态字符串

		<div id="example-4">
		  <input type="radio" id="one" value="One" v-model="picked">
		  <label for="one">One</label>
		  <br>
		  <input type="radio" id="two" value="Two" v-model="picked">
		  <label for="two">Two</label>
		  <br>
		  <span>Picked: {{ picked }}</span>
		</div>

		new Vue({
		  el: '#example-4',
		  data: {
		    picked: ''
		  }
		})

* `input checkbox` - 复选框 - 绑定布尔值（单个框）/数组（多个框）

		<!-- 单框 -->
		<input type="checkbox" id="checkbox" v-model="checked">
		<label for="checkbox">{{ checked }}</label>

		<!-- 多框 -->
		<div id='example-3'>
		  <input type="checkbox" id="jack" value="Jack" v-model="checkedNames">
		  <label for="jack">Jack</label>
		  <input type="checkbox" id="john" value="John" v-model="checkedNames">
		  <label for="john">John</label>
		  <input type="checkbox" id="mike" value="Mike" v-model="checkedNames">
		  <label for="mike">Mike</label>
		  <br>
		  <span>Checked names: {{ checkedNames }}</span>
		</div>

		new Vue({
		  el: '#example-3',
		  data: {
		    checkedNames: []
		  }
		})

* `select` - 选择框 - 绑定静态字符串（单选）/数组（多选）

		<!-- 单选 -->
		<div id="example-5">
		  <select v-model="selected">
		    <option disabled value="">请选择</option>
		    <option>A</option>
		    <option>B</option>
		    <option>C</option>
		  </select>
		  <span>Selected: {{ selected }}</span>
		</div>
		
		new Vue({
		  el: '...',
		  data: {
		    selected: ''
		  }
		})

		<!-- 多选 -->
		<div id="example-6">
		  <select v-model="selected" multiple style="width: 50px;">
		    <option>A</option>
		    <option>B</option>
		    <option>C</option>
		  </select>
		  <br>
		  <span>Selected: {{ selected }}</span>
		</div>
		new Vue({
		  el: '#example-6',
		  data: {
		    selected: []
		  }
		})

**注意：**

* `v-model`会忽略表单元素的`value`，`checked`，`selected`属性的初始值，而总是将Vue实例的数据作为数据来源。故应该通过JavaScript组件的`data`选项声明初始值。
* `v-model`不会在输入法组合文字过程中得到更新。如果要处理该过程，请使用`input`组件。
* 为了在iOS中适用，`select`要提供一个值为空的禁用选项

		<select v-model="selected">
			<option disabled value="">请选择</option>
			<option>A</option>
			<option>B</option>
			<option>C</option>
		</select>

###7.2 值绑定
对于单选按钮，复选框及选择框的选项，v-model 绑定的值通常是静态字符串。

若想把值绑定到 Vue 实例的一个**动态属性**上，这时可以用 `v-bind` 实现，并且这个属性的值**可以不是字符串**。

* `input radio` - 单选按钮

		<input type="radio" v-model="pick" v-bind:value="a">
		// 当选中时
		vm.pick === vm.a

* `input checkbox` - 复选框

		<input
		  type="checkbox"
		  v-model="toggle"
		  true-value="yes"
		  false-value="no"
		>

		// 当选中时
		vm.toggle === 'yes'
		// 当没有选中时
		vm.toggle === 'no'

* `select` - 选择框

		<select v-model="selected">
		    <!-- 内联对象字面量 -->
		  <option v-bind:value="{ number: 123 }">123</option>
		</select>
		
		// 当选中时
		typeof vm.selected // => 'object'
		vm.selected.number // => 123

###7.3 修饰符
* `.lazy` - 将"input"时更新转变为"change"时更新
* `.number` - 自动将用户输入字符转为数值
* `trim` - 自动过滤用户输入的首尾空白字符

###7.4 组件应用
Vue 的组件系统允许你创建具有完全自定义行为且可复用的输入组件。

这些输入组件甚至可以和 v-model 一起使用！

要了解更多，请参阅组件指南中的**自定义输入组件**。

（暂不深入）

##8 组件基础
第 1.5 节已经简单地叙述了组件的注册和使用。这里进行深入地讨论。

###8.1 基本使用
**(1) 组件的基本内容**

组件是可复用的Vue实例，且带有一个名字。

因此它们与 `new Vue` 接收相同的选项，例如 `data`、`computed`、`watch`、`methods` 以及生命周期钩子等。

仅有的例外是像 `el` 这样根实例特有的选项。

基本使用方法参见 **1.5节 组件化**应用。

**(2) 组件的复用**
每用一次组件，就会有一个它的新实例被创建。例如：

	// 定义一个名为 button-counter 的新组件
	Vue.component('button-counter', {
	  data: function () {
	    return {
	      count: 0
	    }
	  },
	  template: '<button v-on:click="count++">You clicked me {{ count }} times.</button>'
	})

	new Vue({ el: '#components-demo' })

	<div id="components-demo">
	  <button-counter></button-counter>
	  <button-counter></button-counter>
	  <button-counter></button-counter>
	</div>

组件的`data`属性必须是一个函数，以此独立维护组件实例的值，否则所有组件共享该变量。

**(3) 组件的组织**

通常一个应用会以一棵嵌套的组件树的形式来组织：

![图片被吞掉了！](vue_components.png)

为了能在模板中使用，这些组件必须先注册以便 Vue 能够识别。

这里有两种组件的注册类型：全局注册和局部注册。至此，我们的组件都只是通过 Vue.component 全局注册的：

	Vue.component('my-component-name', {
	  // ... options ...
	})

全局注册的组件可以用在其被注册之后的任何 (通过 `new Vue`) 新创建的 Vue 根实例，也包括其组件树中的所有子组件的模板中。

**(4) 监听子组件事件**：

Vue实例提供了一个自定义事件的系统来解决子组件向父组件传递数据的问题。

调用内建的 `$emit` 方法并传入事件的名字，来向父级组件传递自定义一个事件。

父级组件可以像处理 native DOM 事件一样通过 `v-on` 监听子组件实例的任意事件。

* **`$emit('事件名', '抛出参数名')`** - 自定义事件，子组件调用并触发该自定义事件
	* 抛出参数名 - 可以指定第二个参数来向上级组件抛出一个值
		* 如果事件处理语句是表达式，在上级组件中用 `$event` 获取
		* 如果事件处理语句是函数，该值会作为第一个参数传入该函数

案例演示：引入可访问性的功能放大博文字号

* (1) 在父组件添加`postFontSize`数据属性来支持功能：

	new Vue({
		el: '#blog-posts-events-demo',
		data: {
			posts: [/* ... */],
			postFontSize: 1
	)}

	<!-- 它在模板中用来控制所有博文的字号 -->
	<div id="blog-posts-events-demo">
	  <div :style="{ fontSize: postFontSize + 'em' }">
	    <blog-post
	      v-for="post in posts"
	      v-bind:key="post.id"
	      v-bind:post="post"
	    ></blog-post>
	  </div>
	</div>

* (2) 在子组件模板中添加添加一个按钮来放大字号，并定义自定义事件：

	Vue.component('blog-post', {
		props: ['post'],
		template: `
			<div class="blog-post">
				<h3>{{ post.title }}</h3>
				<button v-on:click="$emit('enlarge-text', 0.1)">
					Enlarge text
				</button>
				<div v-html="post.content"></div>
			</div>
		`
	})

* (3) 在父组件上监听事件

	<!-- 注：这个标签仍然属于父组件，其里面的内容才是子组件 -->
	<blog-post
		v-for="post in posts",
		v-bind:post="post",
		v-bind:key="post.id",
		v-on:enlarge-text="postFontSize += $event"
	></blog-post>

###8.2 参数传递
向子组件传递数据通过组件选项 `props` 实现。

	// 用一个 props 选项将其包含在该组件可接受的 prop 列表中
	Vue.component('blog-post', {
	  props: ['title'],
	  template: '<h3>{{ title }}</h3>'
	})

`props` 数组内定义的属性会成为该组件的 HTML 元素属性，可以直接在元素内使用并赋值。

	<blog-post title="My journey with Vue"></blog-post>
	<blog-post title="Blogging with Vue"></blog-post>
	<blog-post title="Why Vue is so fun"></blog-post>

当一个值传递给一个 `prop` （注意不是`props`） 属性的时候，它就变成了那个组件实例的一个属性，可以向 `data` 中的值一样访问。

这样，在一般情况下，可以通过 `v-bind` 来动态传递 `prop`。

	new Vue({
	  el: '#blog-post-demo',
	  data: {
	    posts: [
	      { id: 1, title: 'My journey with Vue' },
	      { id: 2, title: 'Blogging with Vue' },
	      { id: 3, title: 'Why Vue is so fun' }
	    ]
	  }
	})
	
	<blog-post
	  v-for="post in posts"
	  v-bind:key="post.id"
	  v-bind:title="post.title"
	></blog-post>

###8.3 使用技巧
**(1) 单个根元素**：

每个组件必须只有一个根元素（即根标签），所以当渲染多个元素时，可以将模板的内容包裹在一个父元素内。

**(2) 列表重构**：

当组件变得复杂、元素增多的时候，可以重构组件，让它接收一个元素变量集合（post）。

	Vue.component('blog-post', {
	  props: ['post'],
	  template: `
		    <div class="blog-post">
		      <h3>{{ post.title }}</h3>
		      <div v-html="post.content"></div>
		    </div>
		`
	})

	<blog-post
	  v-for="post in posts"
	  v-bind:key="post.id"
	  v-bind:post="post"
	></blog-post>

**(3) 使用`v-model`**：

自定义组件支持使用`v-model`，但需自己配置支持`v-model`的接口。

先观察`v-model`的等价形式：

	<input v-model="searchText"/>

等价于：

	<input
		v-bind:value="searchText",
		v-on:input="searchText = $event.target.value" />


所以，编写组件的时候，在模板中相应地实现底层支持：

	Vue.component('custom-input', {
		props: ['value'],
		template: `
			<input
				v-bind:value="value",
				v-on:input="$emit('input', $event.target.value)"
			/>
		`
	})

现在`v-model`就可以在这个组件上完美地工作。

	<custom-input v-model="searchText"></custom-input>

**(4) 通过插槽分发内容 ???**

和HTML元素一样，我们经常需要向一个组件传递内容，Vue自定义的`<slot>`元素让这变得非常简单：

	Vue.component('alert-box', {
		template: `
			<div class="demo-alert-box">
				<strong>Error!</strong>
				<slot></slot>
			</div>
		`
		})

**(5) 动态组件**

有时候需要在不同组件之间进行切换，通过向 Vue 的 `<component>` 元素加一个`is`特性可以实现：

		<!-- 组件会在 `currentTabComponent` 改变时改变 -->
		<component v-bind:is="currentTabComponent"></component>

其中，CurrentTabComponent可以包括：

* 已注册组件的名字
* 一个组件的选项对象

**(6) 解析DOM注意事项**

有些HTML元素，诸如`<ul>`，`<ol>`，`<table>`，`<select>`，对内部元素有严格限制。

使用`is`特性可以让元素具有组件身份，解决这个问题。

要**注意**从以下来源使用模板，这条限制是**不存在**的：

* 字符串（`template: '...'`）
* 单文件组件（`.vue`）
* `<script type="text/x-template">`

##9 过渡 & 动画

##10 可复用性 & 组合
###10.5 过滤器
Vue.js 允许你自定义过滤器，可被用于一些常见的文本格式化。

**定义**

* 过滤器可以用在两个地方：**双花括号插值**和 **`v-bind` 表达式** 。
* 过滤器应该被添加在 JavaScript 表达式的尾部，由“管道”符号指示。

如下所示：

	<!-- 在双花括号中 -->
	{{ message | capitalize }}

	<!-- 在 `v-bind` 中 -->
	<div v-bind:id="rawId | formatId"></div>

**添加过滤器**

可以在一个组件的选项中定义本地的过滤器：

	filters: {
	  capitalize: function (value) {
	    if (!value) return ''
	    value = value.toString()
	    return value.charAt(0).toUpperCase() + value.slice(1)
	  }
	}

或者在创建 Vue 实例**之前**全局定义过滤器：

	Vue.filter('capitalize', function (value) {
	  if (!value) return ''
	  value = value.toString()
	  return value.charAt(0).toUpperCase() + value.slice(1)
	})
	
	new Vue({
	  // ...
	})

**注释：**

* 过滤器函数总接收表达式的值 (之前的操作链的结果) 作为第一个参数。
* 过滤器可以串联：

		{{ message | filterA | filterB }}

* 过滤器是 JavaScript 函数，因此可以接收参数：

		// 这里，filterA 被定义为接收三个参数的过滤器函数。
		// 其中 message 的值作为第一个参数，
		// 普通字符串 'arg1' 作为第二个参数，
		// 表达式 arg2 的值作为第三个参数。
		{{ message | filterA('arg1', arg2) }}

##11 工具

##12 规模化

##13 内在

##14 Cookbook
###14.1 使用 axios 访问 API
有很多时候你在构建应用时需要访问一个 API 并展示其数据。

做这件事的方法有好几种，而使用基于 promise 的 HTTP 客户端 axios 则是其中非常流行的一种。

####14.1.1 安装 axios
使用 npm:

	$ npm install axios

使用 bower:

	$ bower install axios

使用 cdn:

	<script src="https://unpkg.com/axios/dist/axios.min.js"></script>

下面的示例将使用 [CoinDesk API](https://www.coindesk.com/api/) 来完成展示比特币价格且每分钟更新的工作。

####14.1.2 基本示例：查看数据格式
我们有很多种方式可以从 API 请求信息，但是最好首先确认这些数据看起来长什么样，以便进一步确定如何展示它。

确定请求：

	https://api.coindesk.com/v1/bpi/currentprice.json

创建一个 `data` 里的属性以最终放置信息，在 `mounted` 生命周期钩子中获取数据并赋值过去：

	new Vue({
	  el: '#app',
	  data () {
	    return {
	      info: null
	    }
	  },
	  mounted () {
	    axios
	      .get('https://api.coindesk.com/v1/bpi/currentprice.json')
	      .then(response => (this.info = response))
	  }
	})

	<div id="app">
	  {{ info }}
	</div>

结果为：

> { "data": { "time": { "updated": "Feb 19, 2019 07:24:00 UTC", "updatedISO": "2019-02-19T07:24:00+00:00", "updateduk": "Feb 19, 2019 at 07:24 GMT" }, "disclaimer": "This data was produced from the CoinDesk Bitcoin Price Index (USD). Non-USD currency data converted using hourly conversion rate from openexchangerates.org", "chartName": "Bitcoin", "bpi": { "USD": { "code": "USD", "symbol": "&#36;", "rate": "3,925.4300", "description": "United States Dollar", "rate_float": 3925.43 }, "GBP": { "code": "GBP", "symbol": "&pound;", "rate": "3,042.6518", "description": "British Pound Sterling", "rate_float": 3042.6518 }, "EUR": { "code": "EUR", "symbol": "&euro;", "rate": "3,475.0458", "description": "Euro", "rate_float": 3475.0458 } } }, "status": 200, "statusText": "", "headers": { "content-type": "application/javascript", "cache-control": "max-age=15", "expires": "Tue, 19 Feb 2019 07:26:07 UTC" }, "config": { "transformRequest": {}, "transformResponse": {}, "timeout": 0, "xsrfCookieName": "XSRF-TOKEN", "xsrfHeaderName": "X-XSRF-TOKEN", "maxContentLength": -1, "headers": { "Accept": "application/json, text/plain, */*" }, "method": "get", "url": "https://api.coindesk.com/v1/bpi/currentprice.json" }, "request": {} }

####14.1.3 真实示例：和数据协同工作
从结果中发现我们需要的信息在 `response.data.bpi` 中。

为此换用：

	axios
	  .get('https://api.coindesk.com/v1/bpi/currentprice.json')
	  .then(response => (this.info = response.data.bpi))

得到：

> { "USD": { "code": "USD", "symbol": "&#36;", "rate": "3,925.4300", "description": "United States Dollar", "rate_float": 3925.43 }, "GBP": { "code": "GBP", "symbol": "&pound;", "rate": "3,042.6518", "description": "British Pound Sterling", "rate_float": 3042.6518 }, "EUR": { "code": "EUR", "symbol": "&euro;", "rate": "3,475.0458", "description": "Euro", "rate_float": 3475.0458 } }

**建立过滤器**

我们会创建一个过滤器来确保小数部分的合理展示。

	<div id="app">
	  <h1>Bitcoin Price Index</h1>
	  <div
	    v-for="currency in info"
	    class="currency"
	  >
	    {{ currency.description }}:
	    <span class="lighten">
	      <span v-html="currency.symbol"></span>{{ currency.rate_float | currencydecimal }}
	    </span>
	  </div>
	</div>
	
	filters: {
	  currencydecimal (value) {
	    return value.toFixed(2)
	  }
	},

####14.1.4 错误处理
很多时候我们可能并没有从 API 获取想要的数据。

在 axios 中，我们会通过使用 `catch` 来做这件事。

	axios
	  .get('https://api.coindesk.com/v1/bpi/currentprice.json')
	  .then(response => (this.info = response.data.bpi))
	  .catch(error => console.log(error))

为优化用户体验，避免加载时间过长使用户面对空白屏幕等待，构建一个加载效果，然后在根本无法获取数据时通知用户。

	new Vue({
	  el: '#app',
	  data () {
	    return {
	      info: null,
	      loading: true,
	      errored: false
	    }
	  },
	  filters: {
	    currencydecimal (value) {
	      return value.toFixed(2)
	    }
	  },
	  mounted () {
	    axios
	      .get('https://api.coindesk.com/v1/bpi/currentprice.json')
	      .then(response => {
	        this.info = response.data.bpi
	      })
	      .catch(error => {
	        console.log(error)
	        this.errored = true
	      })
	      .finally(() => this.loading = false)
	  }
	})

	<div id="app">
	  <h1>Bitcoin Price Index</h1>
	
	  <section v-if="errored">
	    <p>We're sorry, we're not able to retrieve this information at the moment, please try back later</p>
	  </section>
	
	  <section v-else>
	    <div v-if="loading">Loading...</div>
	
	    <div
	      v-else
	      v-for="currency in info"
	      class="currency"
	    >
	      {{ currency.description }}:
	      <span class="lighten">
	        <span v-html="currency.symbol"></span>{{ currency.rate_float | currencydecimal }}
	      </span>
	    </div>
	
	  </section>
	</div>