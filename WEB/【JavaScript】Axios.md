##目录

##1 安装
使用 npm:

	$ npm install axios

使用 bower:

	$ bower install axios

使用 cdn:

	<script src="https://unpkg.com/axios/dist/axios.min.js"></script>

##2 语法样例
###2.1 执行 `GET` 请求

	// 为给定 ID 的 user 创建请求
	axios.get('/user?ID=12345')
	  .then(function (response) {
	    console.log(response);
	  })
	  .catch(function (error) {
	    console.log(error);
	  });
	
	// 可选地，上面的请求可以这样做
	axios.get('/user', {
	    params: {
	      ID: 12345
	    }
	  })
	  .then(function (response) {
	    console.log(response);
	  })
	  .catch(function (error) {
	    console.log(error);
	  });

###2.2 执行 `POST` 请求

	axios.post('/user', {
	    firstName: 'Fred',
	    lastName: 'Flintstone'
	  })
	  .then(function (response) {
	    console.log(response);
	  })
	  .catch(function (error) {
	    console.log(error);
	  });

###2.3 执行多个并发请求

	function getUserAccount() {
	  return axios.get('/user/12345');
	}
	
	function getUserPermissions() {
	  return axios.get('/user/12345/permissions');
	}
	
	axios.all([getUserAccount(), getUserPermissions()])
	  .then(axios.spread(function (acct, perms) {
	    // 两个请求现在都执行完成
	  }));

##3 请求
###3.1 请求方法 `axios()`
* `axios(config)`

发起一个POST请求：

	axios({
	  method: 'post',
	  url: '/user/12345',
	  data: {
	    firstName: 'Fred',
	    lastName: 'Flintstone'
	  }
	});

获取远程图片：

	axios({
	  method:'get',
	  url:'http://bit.ly/2mTM3nY',
	  responseType:'stream'
	})
	  .then(function(response) {
	  response.data.pipe(fs.createWriteStream('ada_lovelace.jpg'))
	});

* `axios(url[, config])`

发起一个GET请求（GET是默认的请求方法）:

	axios('/user/12345');

###3.2 请求方法别名
####3.2.1 方法
* `axios.request(config)`
* `axios.get(url[, config])`
* `axios.delete(url[, config])`
* `axios.head(url[, config])`
* `axios.options(url[, config])`
* `axios.post(url[, data[, config]])`
* `axios.put(url[, data[, config]])`
* `axios.patch(url[, data[, config]])`

**注释：**

当使用以上别名方法时，`url`，`method` 和 `data` 等属性不用在 `config` 重复声明。

####3.2.2 配置
下面是所有可用的请求配置项，只有 `url` 是必填，默认的请求方法是 `GET`，如果没有指定请求方法的话。

	{
	  // `url` 是请求的接口地址
	  url: '/user',
	
	  // `method` 是请求的方法
	  method: 'get', // 默认值
	
	  // 如果url不是绝对路径，那么会将baseURL和url拼接作为请求的接口地址
	  // 用来区分不同环境，建议使用
	  baseURL: 'https://some-domain.com/api/',
	
	  // 用于请求之前对请求数据进行操作
	  // 只用当请求方法为‘PUT’，‘POST’和‘PATCH’时可用
	  // 最后一个函数需return出相应数据
	  // 可以修改headers
	  transformRequest: [function (data, headers) {
	    // 可以对data做任何操作
	
	    return data;
	  }],
	
	  // 用于对相应数据进行处理
	  // 它会通过then或者catch
	  transformResponse: [function (data) {
	    // 可以对data做任何操作
	
	    return data;
	  }],
	
	  // `headers` are custom headers to be sent
	  headers: {'X-Requested-With': 'XMLHttpRequest'},
	
	  // URL参数
	  // 必须是一个纯对象或者 URL参数对象
	  params: {
	    ID: 12345
	  },
	
	  // 是一个可选的函数负责序列化`params`
	  // (e.g. https://www.npmjs.com/package/qs, http://api.jquery.com/jquery.param/)
	  paramsSerializer: function(params) {
	    return Qs.stringify(params, {arrayFormat: 'brackets'})
	  },
	
	  // 请求体数据
	  // 只有当请求方法为'PUT', 'POST',和'PATCH'时可用
	  // 当没有设置`transformRequest`时，必须是以下几种格式
	  // - string, plain object, ArrayBuffer, ArrayBufferView, URLSearchParams
	  // - Browser only: FormData, File, Blob
	  // - Node only: Stream, Buffer
	  data: {
	    firstName: 'Fred'
	  },
	
	  // 请求超时时间（毫秒）
	  timeout: 1000,
	
	  // 是否携带cookie信息
	  withCredentials: false, // default
	
	  // 统一处理request让测试更加容易
	  // 返回一个promise并提供一个可用的response
	  // 其实我并不知道这个是干嘛的！！！！
	  // (see lib/adapters/README.md).
	  adapter: function (config) {
	    /* ... */
	  },
	
	  // `auth` indicates that HTTP Basic auth should be used, and supplies credentials.
	  // This will set an `Authorization` header, overwriting any existing
	  // `Authorization` custom headers you have set using `headers`.
	  auth: {
	    username: 'janedoe',
	    password: 's00pers3cret'
	  },
	
	  // 响应格式
	  // 可选项 'arraybuffer', 'blob', 'document', 'json', 'text', 'stream'
	  responseType: 'json', // 默认值是json
	
	  // `xsrfCookieName` is the name of the cookie to use as a value for xsrf token
	  xsrfCookieName: 'XSRF-TOKEN', // default
	
	  // `xsrfHeaderName` is the name of the http header that carries the xsrf token value
	  xsrfHeaderName: 'X-XSRF-TOKEN', // default
	
	  // 处理上传进度事件
	  onUploadProgress: function (progressEvent) {
	    // Do whatever you want with the native progress event
	  },
	
	  // 处理下载进度事件
	  onDownloadProgress: function (progressEvent) {
	    // Do whatever you want with the native progress event
	  },
	
	  // 设置http响应内容的最大长度
	  maxContentLength: 2000,
	
	  // 定义可获得的http响应状态码
	  // return true、设置为null或者undefined，promise将resolved,否则将rejected
	  validateStatus: function (status) {
	    return status >= 200 && status < 300; // default
	  },
	
	  // `maxRedirects` defines the maximum number of redirects to follow in node.js.
	  // If set to 0, no redirects will be followed.
	  // 最大重定向次数？没用过不清楚
	  maxRedirects: 5, // default
	
	  // `httpAgent` and `httpsAgent` define a custom agent to be used when performing http
	  // and https requests, respectively, in node.js. This allows options to be added like
	  // `keepAlive` that are not enabled by default.
	  httpAgent: new http.Agent({ keepAlive: true }),
	  httpsAgent: new https.Agent({ keepAlive: true }),
	
	  // 'proxy' defines the hostname and port of the proxy server
	  // Use `false` to disable proxies, ignoring environment variables.
	  // `auth` indicates that HTTP Basic auth should be used to connect to the proxy, and
	  // supplies credentials.
	  // This will set an `Proxy-Authorization` header, overwriting any existing
	  // `Proxy-Authorization` custom headers you have set using `headers`.
	  // 代理
	  proxy: {
	    host: '127.0.0.1',
	    port: 9000,
	    auth: {
	      username: 'mikeymike',
	      password: 'rapunz3l'
	    }
	  },
	
	  // `cancelToken` specifies a cancel token that can be used to cancel the request
	  // (see Cancellation section below for details)
	  // 用于取消请求？又是一个不知道怎么用的配置项
	  cancelToken: new CancelToken(function (cancel) {
	  })
	}

##4 响应
###4.1 响应处理
某个请求的响应包含以下信息：

	{
	  // `data` 由服务器提供的响应
	  data: {},
	
	  // `status` 来自服务器响应的 HTTP 状态码
	  status: 200,
	
	  // `statusText` 来自服务器响应的 HTTP 状态信息
	  statusText: 'OK',
	
	  // `headers` 服务器响应的头
	  headers: {},
	
	  // `config` 是为请求提供的配置信息
	  config: {}
	}

使用 then 时，你将接收下面这样的响应：

	axios.get('/user/12345')
	  .then(function(response) {
	    console.log(response.data);
	    console.log(response.status);
	    console.log(response.statusText);
	    console.log(response.headers);
	    console.log(response.config);
	  });

###4.2 错误处理
在使用 catch 时，或传递 rejection callback 作为 then 的第二个参数时，响应可以通过 error 对象可被使用。

	axios.get('/user/12345')
	  .catch(function (error) {
	    if (error.response) {
	      // 请求已发出，但服务器响应的状态码不在 2xx 范围内
	      console.log(error.response.data);
	      console.log(error.response.status);
	      console.log(error.response.headers);
	    } else {
	      // Something happened in setting up the request that triggered an Error
	      console.log('Error', error.message);
	    }
	    console.log(error.config);
	  });

可以使用 validateStatus 配置选项定义一个自定义 HTTP 状态码的错误范围。

	axios.get('/user/12345', {
	  validateStatus: function (status) {
	    return status < 500; // 状态码在大于或等于500时才会 reject
	  }
	})

