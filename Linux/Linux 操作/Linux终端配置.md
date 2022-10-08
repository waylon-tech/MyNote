## 目录

[toc]

## 终端配置文件

首先看看相关的配置文件有哪些：

*  **`/etc/profile`**：此文件为系统的每个用户设置环境信息，当用户第一次登录时，该文件被执行，并从 `/etc /profile.d` 目录的配置文件中搜集 shell 的设置。此文件默认调用 **`/etc/bash.bashrc`** 文件

*  **`/etc/bashrc`**：为每一个运行 bash shell 的用户执行此文件。当 bash shell 被打开时，该文件被读取

* **`~/.bash_profile`**：每个用户都可使用该文件输入专用于自己使用的 shell 信息，当用户登录时，该文件仅仅执行一次！默认情况下，它设置一些环境变量，然后执行用户的 `.bashrc` 文件

* **`~/.bashrc`**：该文件包含专用于你的 bash shell 的 bash 信息

* **`~/.bash_logout`**：当每次退出系统（退出 bash shell）时，执行该文件

## 颜色配置

对 **`～/.bashrc`** 文件中的 **`PS1`** 变量进行定制。

**`PS1` 变量代表的内容**：`用户名+主机名+路径名（长路径）+ $`

**颜色的设置公式**：`颜色=\033[代码;前景;背景m]`，其中中括号 `[` 和 `]` 需要用反斜杠 `\` 转义。

其的含义为：

| 前景 | 背景 | 颜色   |
| ---- | ---- | ------ |
| 30   | 40   | 黑色   |
| 31   | 41   | 红色   |
| 32   | 42   | 绿色   |
| 33   | 43   | 黄色   |
| 34   | 44   | 蓝色   |
| 35   | 45   | 紫红色 |
| 36   | 46   | 青蓝色 |
| 37   | 47   | 白色   |
|      | 1    | 透明色 |

| 代码 | 意义      |
| ---- | --------- |
| 0    | OFF       |
| 1    | 高亮显示  |
| 4    | underline |
| 5    | 闪烁      |
| 7    | 反白显示  |
| 8    | 不可见    |

## 配置保存

使用 `source ~/.bashrc` 就可以激活配置文件了。

注意，如果每次登陆终端不能读取配置文件，需要对 `/etc/profile` 文件进行更改：

```shell
if [ "${PS1-}" ]; then
  if [ "${BASH-}" ] && [ "$BASH" != "/bin/sh" ]; then
    # The file bash.bashrc already sets the default PS1.
    # PS1='\h:\w\$ '
    if [ -f /etc/bash.bashrc ]; then
      . /etc/bash.bashrc
      . ~/.bashrc	# ！！！加入这行！！！
    fi
  else
    if [ "`id -u`" -eq 0 ]; then
      PS1='# '
    else
      PS1='$ '
    fi
  fi
fi
```

## 配置命令 `ls`

最后，可以对最常用的命令 `ls` 进行设置，通过在 `.bashrc` 文件中设置 `alias` 以实现 `ls` 命令的一些自定义设置：

```shell
# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ls="ls --color=auto"
```

