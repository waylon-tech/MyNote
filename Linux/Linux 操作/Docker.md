## 目录

[toc]

## 1 介绍

Docker 是一个开源的应用容器引擎，基于 [Go 语言](https://www.runoob.com/go/go-tutorial.html) 并遵从 Apache 2.0 协议开源。

Docker 可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。

容器是完全使用沙箱机制，相互之间不会有任何接口（类似 iPhone 的 app）,更重要的是容器性能开销极低。

Docker 官网：[https://www.docker.com](https://www.docker.com/)

Github Docker 源码：https://github.com/docker/docker-ce

## 2 Docker 架构

### 2.1 理解

docker 思想有三个：

1. 集装箱：所有需要的内容会放到集装箱中，谁需要这些环境就直接拿这个集装箱就可以了

2. 标准化：

   * 运输的标准化：Docker 有一个码头，所有上传的集装箱都放在了这个码头上，当谁需要某一个环境，就直接指派大海豚去搬运这个集装箱

   * 命令的标准化：Docker 提供一系列的命令，帮助我们去获取集装箱等等操作

   * 提供了 REST 的 API：衍生出了很多图形化界面，Rancher

3. 隔离性：Docker 在运行集装箱的内容时，会在 Linux 的内核中，单独开辟一片空间，这片空间不会影响其他的程序

### 2.2 概念

基本概念:

- **镜像（Image）**：Docker 镜像（Image），就相当于是一个 root 文件系统。比如官方镜像 `ubuntu:16.04` 就包含了完整的一套 `Ubuntu16.04` 最小系统的 root 文件系统。
- **容器（Container）**：镜像（Image）和容器（Container）的关系，就像是面向对象程序设计中的类和实例一样，镜像是静态的定义，容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。
- **仓库（Repository）**：仓库可看成一个代码控制中心，用来保存镜像，分为公有仓库和私有仓库；每个仓库可以包含多个标签（Tag，用于版本管理）；每个标签对应一个镜像

其他概念：

* 客户端（Client）：通过命令行或者其他工具使用 [Docker SDK](https://docs.docker.com/develop/sdk/) 与 Docker 的守护进程通信
* Registry：Docker Registry 中可以包含多个仓库（Repository）
  * [Docker Hub](https://hub.docker.com) 提供了庞大的镜像集合供使用，是顶级 Registry
* Machine：一个简化 Docker 安装的命令行工具，通过一个简单的命令行即可在相应的平台上安装 Docker

Docker 使用客户端-服务器 (C/S) 架构模式，使用远程 API 来管理和创建 Docker 容器：

![img](img/576507-docker1.png)

## 3 Docker 安装

暂略，用到时再补。

参见系列文章：https://www.runoob.com/docker/ubuntu-docker-install.html

* 开关 docker

  ```shell
  sudo systemctl enable docker
  sudo systemctl start docker
  ```

* 验证是否安装成功

  ```shell
  sudo docker run hello-world
  ```

* docker 权限管理

  【注】`docker` 使用组给予的权限相当于 `root`，会降低计算机的安全性，请慎重考虑。

  ```shell
  # 创建 docker 用户组
  sudo groupadd docker
  
  # 添加用户到 docker 用户组
  sudo usermod -aG docker $USER
  
  # 激活改动
  newgrp docker
  
  # 验证
  docker run hello-world
  ```

  错误处理：如果在未加入权限时，用 `sudo` 验证过，可能会有以下错误

  ```shell
  WARNING: Error loading config file: /home/user/.docker/config.json -
  stat /home/user/.docker/config.json: permission denied
  ```

  只需要删除 `~/.docker/` 目录（存储个性化信息）或更改权限即可：

  ```shell
  sudo chown "$USER":"$USER" /home/"$USER"/.docker -R
  sudo chmod g+rwx "$HOME/.docker" -R
  ```

## 4 Docker 使用
### 4.0 帮助
* `docker version`：显示 docker 版本信息
* `docker info`：显示 docker 的系统信息
* `docker [命令] —help`：帮助命令

最后，可以参考官网的帮助文档

### 4.1 镜像

#### 4.1.1 列出镜像

* `docker images [OPTIONS]`：列出本地镜像
  * `a`：列出所有镜像
  * `q`：只显示镜像 `id`

  ```shell
  w3cschool@w3cschool:~$ docker images
  REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
  ubuntu              14.04               90d5884b1ee0        5 days ago          188 MB
  nginx               latest              6f8d099c3adc        12 days ago         182.7 MB
  mysql               5.6                 f2e8d6c772c0        3 weeks ago         324.6 MB
  httpd               latest              02ef73cf1bc0        3 weeks ago         194.4 MB
  ubuntu              15.10               4e3b13c8a266        4 weeks ago         136.3 MB
  ```

  * `REPOSTITORY`：表示镜像的仓库源
  * `TAG`：镜像的标签
    * 同一仓库源可以有多个 `TAG`，代表这个仓库源的不同个版本，如那个 ubuntu
    * 后面在建立容器是，通过 `REPOSTITORY:TAG` 来定位哪个仓库里的哪个镜像
  * `IMAGE ID`：镜像ID
  * `CREATED`：镜像创建时间
  * `SIZE`：镜像大小

#### 4.1.2 查找镜像

* `docker search [OPTIONS] TERM`：从 Docker Hub 查找镜像

  * `OPTIONS`
    * `--automated`：只列出 automated build 类型的镜像
    * `--no-trunc`：显示完整的镜像描述
    * `-f <过滤条件>`：列出收藏数不小于指定值的镜像
  * `TREM`：查找的镜像名

  ```shell
  runoob@runoob:~$ docker search -f stars=10 java
  NAME                  DESCRIPTION                           STARS   OFFICIAL   AUTOMATED
  java                  Java is a concurrent, class-based...   1037    [OK]       
  anapsix/alpine-java   Oracle Java 8 (and 7) with GLIBC ...   115                [OK]
  develar/java                                                 46                 [OK]
  isuper/java-oracle    This repository contains all java...   38                 [OK]
  lwieske/java-8        Oracle Java 8 Container - Full + ...   27                 [OK]
  nimmis/java-centos    This is docker images of CentOS 7...   13                 [OK]
  ```

  * `NAME`：镜像仓库源的名称
  * `DESCRIPTION`：镜像的描述
  * `STARS`：类似 Github 里面的 star，表示点赞、喜欢的意思
  * `OFFICIAL`：是否 docker 官方发布
  * `AUTOMATED`：自动构建

#### 4.1.3 获取镜像

* `docker pull 镜像名称[:tag]` - 拉取镜像到本地

  ```shell
  Cw3cschool@w3cschool:~$ docker pull ubuntu:13.10
  13.10: Pulling from library/ubuntu  # 如果不写 tag，会默认使用最新的
  6599cadaf950: Pull complete  # 分层下载，docker image 的核心，涉及联合文件系统
  23eda618d451: Pull complete 
  f0be3084efe9: Pull complete 
  52de432f084b: Pull complete 
  a3ed95caeb02: Pull complete 
  Digest: sha256:15b79a6654811c8d992ebacdfbd5152fcf3d165e374e264076aa435214a947a3  # 签名
  Status: Downloaded newer image for ubuntu:13.10
  # 这里可能还有一行，表示镜像的真实地址
  ```

  * docker 先在本地上搜索，如果不存在就会到 docker 镜像仓库中搜索下载

#### 4.1.4 删除镜像
* `docker rmi [OPTIONS] id/image1 id/image2 ...`：删除镜像
  * `[OPTIONS]`
  * `-f`：强制删除镜像
  
  ```shell
  Cw3cschool@w3cschool:~$ docker rmi -f $(docker images -aq)
  ```

【注】[修改dcoker镜像和容器存储的位置](https://developer.aliyun.com/article/637851?spm=a2c6h.13813017.0.dArticle738638.3dc8cef7Q4utT7)

#### 4.1.5 构建镜像

需要创建一个` Dockerfile` 文件，其中包含一组指令来告诉 Docker 如何构建我们的镜像。

```shell
w3cschool@w3cschool:~$ cat Dockerfile 
FROM    centos:6.7
MAINTAINER      Fisher "fisher@sudops.com"

RUN     /bin/echo 'root:123456' |chpasswd
RUN     useradd youj
RUN     /bin/echo 'youj:123456' |chpasswd
RUN     /bin/echo -e "LANG=\"en_US.UTF-8\"" &gt; /etc/default/local
EXPOSE  22
EXPOSE  80
CMD     /usr/sbin/sshd -D
```

每一个指令都会在镜像上创建一个新的层，每一个指令的前缀都必须是大写的。

第一条 `FROM`，指定使用哪个镜像源，

`RUN` 指令告诉 docker 在镜像内执行命令，安装了什么。。。

然后，我们使用 `Dockerfile` 文件，通过 `docker build` 命令来构建一个镜像。

```shell
w3cschool@w3cschool:~$ docker build -t youj/centos:6.7 .
Sending build context to Docker daemon 17.92 kB
Step 1 : FROM centos:6.7
 ---&gt; d95b5ca17cc3
Step 2 : MAINTAINER Fisher "fisher@sudops.com"
 ---&gt; Using cache
 ---&gt; 0c92299c6f03
Step 3 : RUN /bin/echo 'root:123456' |chpasswd
 ---&gt; Using cache
 ---&gt; 0397ce2fbd0a
Step 4 : RUN useradd youj
......

# 参数说明：
# -t: 指定要创建的目标镜像名
# .: Dockerfile 文件所在目录，可以指定 Dockerfile 的绝对路径
```

#### 4.1.6 更新镜像

先使用镜像来创建一个容器：

```shell
w3cschool@w3cschool:~$ docker run -it ubuntu:15.10 /bin/bash
root@e218edb10161:/# 

# 参数说明：
# -i: 交互式操作
# -t: 终端
# ubuntu:15.10: ubuntu 镜像，tag 为 15.10
# /bin/bash：放在镜像名后的是命令，这里我们希望有个交互式 Shell，因此用的是 /bin/bash
```

在运行的容器内使用 `apt-get update` 命令进行更新。

在完成操作之后，输入 `exit` 命令来退出这个容器。

此时 ID 为 `e218edb10161` 的容器，是按我们的需求更改的容器。我们可以通过命令 `docker commit` 来提交容器副本。

```shell
w3cschool@w3cschool:~$ docker commit -m="has update" -a="youj" e218edb10161 w3cschool/ubuntu:v2
sha256:70bf1840fd7c0d2d8ef0a42a817eb29f854c1af8f7c59fc03ac7bdee9545aff8

# 参数说明：
# -m: 提交的描述信息
# -a: 指定镜像作者
# e218edb10161: 容器ID
# w3cschool/ubuntu:v2: 指定要创建的目标镜像
```

#### 4.1.7 设置镜像标签

* `docker tag [OPTIONS] IMAGE[:TAG] [REGISTRYHOST/][USERNAME/]NAME[:TAG]`：标记本地镜像，将其归入某一仓库

  ```shell
  # 将镜像ubuntu:15.10 标记为 runoob/ubuntu:v3 镜像
  root@runoob:~$ docker tag ubuntu:15.10 runoob/ubuntu:v3
  root@runoob:~$ docker images   runoob/ubuntu:v3
  REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
  runoob/ubuntu       v3                  4e3b13c8a266        3 months ago        136.3 MB
  
  # 为镜像添加一个新的标签
  w3cschool@w3cschool:~$ docker tag 860c279d2fec youj/centos:dev
  ```

### 4.2 容器
#### 4.2.1  运行容器

* `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`：创建一个新的容器并运行一个命令

  * `OPTIONS`
    - `-a stdin`：指定标准输入输出内容类型，可选 `STDIN`/`STDOUT`/`STDERR` 三项
    - **`-d`**：后台运行容器，并返回容器 ID；
    - **`-i`**：以交互模式运行容器，通常与 `-t` 同时使用；
    - `-P`：随机端口映射，容器内部端口**随机**映射到主机的端口
    - `-p`：指定端口映射，格式为：`主机(宿主)端口:容器端口`，`容器端口`
    - **`-t`**：为容器重新分配一个伪输入终端，通常与 `-i` 同时使用
    - **`—-name=“nginx-lb”`：**：为容器指定一个名称
    - `--dns 8.8.8.8`：指定容器使用的 DNS 服务器，默认和宿主一致
    - `--dns-search example.com`：指定容器 DNS 搜索域名，默认和宿主一致
    - `-h "mars"`：指定容器的 hostname
    - `-e username="ritchie"`：设置环境变量
    - `--env-file=[]`：从指定文件读入环境变量
    - `--cpuset="0-2" or --cpuset="0,1,2"`：绑定容器到指定 CPU 运行
    - `-m`：设置容器使用内存最大值
    - `--net="bridge"`：指定容器的网络连接类型，支持 `bridge`/`host`/`none`/`container`: 四种类型
    - `--link=[]`：添加链接到另一个容器
    - `--expose=[]`：开放一个端口或一组端口
    - `--volume , -v`：绑定一个卷
  * `IMAGE`：要运行的镜像
  * `COMMAND`：容器内运行的命令
  * `ARG`：容器内运行的命令的参数

  ```shell
  # 简单操作
  docker run 镜像ID|镜像名称[:tag]
  
  # 常用参数
  docker run -d -p 宿主机端口:容器端口 --name 镜像ID|镜像名称[:tag]
  # -d: 代表后台运行容器
  # -p: 宿主机端口:容器端口：为了映射当前 Linux 的端口和容器的端口
  # --name: 指定容器名称
  
  # 交互操作
  docker run -it 镜像ID|镜像名称[:tag] /bin/bash
  >> exit # 退出容器（容器会停止）
  >> Ctrl + P + Q # 退出容器（容器不会停止）
  ```
  
  * 其他参数查询方法：`docker run --help | grep -i gpus`（例子，查询 `--gpus` 参数）

【注】当运行容器时，使用的镜像如果在本地中不存在，docker 就会自动从 docker 镜像仓库中下载，默认是从 Docker Hub 公共镜像源下载。

#### 4.2.2 删除容器

* `docker rm [OPTIONS] 容器id`：删除容器
  * `OPTIONS`
  * `-f`：强制删除
  
  ```shell
  w3cschool@w3cschool:~$ docker rm -f $(docker ps -aq)
  ```

#### 4.2.3 查看容器

* `docker ps [-qa]`：查看正在运行的容器

  * `-a`：查看全部的容器，包括没有运行的
  * `-q`：只查看容器的 ID

  ```shell
  runoob@runoob:~$ docker ps
  CONTAINER ID   IMAGE          COMMAND                ...  PORTS                    NAMES
  09b93464c2f7   nginx:latest   "nginx -g 'daemon off" ...  80/tcp, 443/tcp          myrunoob
  96f7f14e99ab   mysql:5.6      "docker-entrypoint.sh" ...  0.0.0.0:3306->3306/tcp   mymysql
  ```

  * `CONTAINER ID`：容器 ID
  * `IMAGE`：使用的镜像
  * `COMMAND`：启动容器时运行的命令
  * `CREATED`：容器的创建时间
  * `STATUS`：容器状态。状态有7种：
    - `created`（已创建）
    - `restarting`（重启中）
    - `running`（运行中）
    - `removing`（迁移中）
    - `paused`（暂停）
    - `exited`（停止）
    - `dead`（死亡）
  * `PORTS`：容器的端口信息和使用的连接类型（tcp\udp）
  * `NAMES`：自动分配的容器名称

#### 4.2.4 查看日志

* `docker logs -f 容器ID或名字`：查看容器内部的标准输出
  * `-f`：可以滚动查看日志的最后几行

#### 4.2.5 查看容器内进程

* `docker top [OPTIONS] CONTAINER [ps OPTIONS]`：查看容器中运行的进程信息，支持 ps 命令参数

  ```shell
  runoob@runoob:~/mysql$ docker top mymysql
  UID    PID    PPID    C      STIME   TTY  TIME       CMD
  999    40347  40331   18     00:58   ?    00:00:02   mysqld
  ```

#### 4.2.6 检查容器

* `docker inspect [OPTIONS] NAME|ID [NAME|ID...]`：获取容器/镜像的元数据
  * `OPTIONS`
    * `-f`：指定返回值的模板文件
    * `-s`：显示总的文件大小
    * `--type`：为指定类型返回 JSON
  * `NAME`：容器名或镜像名

#### 4.2.7 状态操作

* `docker start [OPTIONS] CONTAINER [CONTAINER...]`：启动已被停止的容器
* `docker stop [OPTIONS] CONTAINER [CONTAINER...]`：停止运行中的容器
* `docker restart [OPTIONS] CONTAINER [CONTAINER...]`：重启容器