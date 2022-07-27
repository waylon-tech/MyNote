##目录

##1 安装
官网示例了解一下：

[http://pandao.github.io/editor.md/examples/index.html](http://pandao.github.io/editor.md/examples/index.html)

下载插件：

[http://pandao.github.io/editor.md/](http://pandao.github.io/editor.md/)

##2 开始使用
引用：

	<link rel="stylesheet" href="lib/js/editor.md-master/css/editormd.css" />
	//依赖jquery
	<script type="text/javascript" src="lib/js/jquery.min.js"></script>
	<script src="lib/js/editor.md-master/editormd.min.js"></script>
	//需要这三个文件，自己对应好目录哦

在页面中添加对应 `id`：

	<div id="layout" class="editor">
	    <div id="test-editormd">
	        <textarea></textarea>
	    </div>
	</div>

然后使用 JavaScript 初始化：

	var testEditor;
	testEditor = editormd("test-editormd", {
	     placeholder:'本编辑器支持Markdown编辑，左边编写，右边预览',  //默认显示的文字，这里就不解释了
	     width: "90%",
	     height: 640,
	     syncScrolling: "single",  
	     path: "lib/js/editor.md-master/lib/",   //你的path路径（原资源文件中lib包在我们项目中所放的位置）
	     theme: "dark",//工具栏主题
	     previewTheme: "dark",//预览主题
	     editorTheme: "pastel-on-dark",//编辑主题
	     saveHTMLToTextarea: true,
	     emoji: false,
	     taskList: true, 
	     tocm: true,         // Using [TOCM]
	     tex: true,                   // 开启科学公式TeX语言支持，默认关闭
	     flowChart: true,             // 开启流程图支持，默认关闭
	     sequenceDiagram: true,       // 开启时序/序列图支持，默认关闭,
	     toolbarIcons : function() {  //自定义工具栏，后面有详细介绍
	         return editormd.toolbarModes['simple']; // full, simple, mini
	      },
	});
	//上面的挑有用的写上去就行

综上所述一个编辑器就诞生了，下面有个小知识点。

	testEditor.getMarkdown();
	// 在js中调用 getMarkdown() 这个方法可以获得 Markdown 格式的文本。

##3 页面展示
后台给我们的文档我们要展示成转换后的样子不能一大堆符号摆在页面上是吧，也不好看呀，所以下面放上展示代码需要的东东。

先引入必要的 JavaScript 文件：

	<link rel="stylesheet" href="lib/js/editor.md-master/css/editormd.css" />
	<link rel="shortcut icon" href="https://pandao.github.io/editor.md/favicon.ico" type="image/x-icon" />
	<script src="lib/js/jquery.min.js""></script>
	<script src="lib/js/editor.md-master/lib/marked.min.js"></script>
	<script src="lib/js/editor.md-master/lib/prettify.min.js"></script>
	<script src="lib/js/editor.md-master/lib/raphael.min.js"></script>
	<script src="lib/js/editor.md-master/lib/underscore.min.js"></script>
	<script src="lib/js/editor.md-master/lib/sequence-diagram.min.js"></script>
	<script src="lib/js/editor.md-master/lib/flowchart.min.js"></script>
	<script src="lib/js/editor.md-master/editormd.min.js"></script>
	//具体目录在你下载的文件里都能找到，对号入座

在页面中添加对应 `id`：

	 <div id="layout"  class="editor">
	    <div id="test-editormd" >
	        <textarea></textarea>
	    </div>
	 </div>

然后使用 JavaScript 初始化：

	testEditor = editormd.markdownToHTML("test-editormd", {
	      markdown:$scope.apidetails.content,
	      htmlDecode      : "style,script,iframe",  // you can filter tags decode
	      emoji           : true,
	      taskList        : true,
	      tex             : true,  // 默认不解析
	      flowChart       : true,  // 默认不解析
	      sequenceDiagram : true,  // 默认不解析
	});

##4 自定义工具栏
工具栏分为三组，full, simple, mini这三个，可以选择，如果想更加自由选你所需，就可以用下面的代码，也可以看看官网的示例。

	toolbarIcons : function() {
	            // Or return editormd.toolbarModes[name]; // full, simple, mini
	            // Using "||" set icons align right.
	            return ["undo", "redo", "|", "bold", "hr"]
	        },

具体没一个标签代表的什么含义可以对照整个工具栏自己对一下，下面是他的源码：

	t.toolbarModes={
	    full:["undo","redo","|","bold","del","italic","quote","ucwords","uppercase","lowercase","|","h1","h2","h3","h4","h5","h6","|","list-ul","list-ol","hr","|","link","reference-link","image","code","preformatted-text","code-block","table","datetime","emoji","html-entities","pagebreak","|","goto-line","watch","preview","fullscreen","clear","search","|","help","info"],
	    simple:["undo","redo","|","bold","del","italic","quote","uppercase","lowercase","|","h1","h2","h3","h4","h5","h6","|","list-ul","list-ol","hr","|","watch","preview","fullscreen","|","help","info"],
	    mini:["undo","redo","|","watch","preview","|","help","info"]
	}