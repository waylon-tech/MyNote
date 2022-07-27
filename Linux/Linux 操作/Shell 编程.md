## 目录

[toc]

## 2 其他

### 2.1 Conda

* 基本操作

  * 升级全部库：`conda upgrade --all`
  * 升级一个包：`conda update packagename`
  * 安装一个包：`conda install packagename`
  * 安装多个包：`conda installl numpy pandas scipy`
  * 安装固定包：`conda install numpy =1.10`
  * 移除一个包：`conda remove packagename`
  * 查看所有包：`conda list`

* 管理环境

  * 创建环境：`conda create -n env_name list of packagenaem`

    `eg:  conda create -n env_name pandas`

  * 指定 python 版本：`conda create -n env_name python2=2.7 pandas` 

  * 激活环境：`activate env_name`

  * 退出环境：`deactivate env_name`

  * 删除环境：`conda env remove -n env_name`

  * 显示所有环境：`conda env list`

### 2.2 VSCode

#### 2.2.1 预定义变量

- `${workspaceFolder}` - 当前工作目录(根目录)

  ```shell
  /home/your-username/your-project
  ```

- `${workspaceFolderBasename}` - 当前文件的父目录

  ```shell
  your-project
  ```

- `${file}` - 当前打开的文件名(完整路径)

  ```shell
  /home/your-username/your-project/folder/file.ext
  ```

- `${relativeFile}` - 当前根目录到当前打开文件的相对路径(包括文件名)

  ```shell
  folder/file.ext

- `${relativeFileDirname}` - 当前根目录到当前打开文件的相对路径(不包括文件名)

  ```shell
  folder

- `${fileBasename}` - 当前打开的文件名(包括扩展名)

  ```shell
  file.ext

- `${fileBasenameNoExtension}` - 当前打开的文件名(不包括扩展名)

  ```shell
  file

- `${fileDirname}` - 当前打开文件的目录

  ```shell
  home/your-username/your-project/folder

- `${fileExtname}` - 当前打开文件的扩展名

  ```shell
  .ext

- `${cwd}` - 启动时task工作的目录

- `${lineNumber}` - 当前激活文件所选行

- `${selectedText}` - 当前激活文件中所选择的文本

- `${execPath}` - `vscode` 执行文件所在的目录

- `${defaultBuildTask}` - 默认编译任务 (build task) 的名字

【注】VSCode 的智能提示会在 `tasks.json` 和 `launch.json` 提示所有支持的预定义变量。
