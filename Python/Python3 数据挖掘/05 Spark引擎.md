## 目录

[toc]

## 1 安装与配置

到[官网](https://www.apache.org/)寻找并下载 Apache Spark tar 文件（没有安装 hadoop 的可以下载带 hadoop 的版本），并解压到指定目录。

```shell
tar -xzvf /home/例如/spark-2.4.1-bin-hadoop2.7.tgz
```

【注】官网的 spark 已经整合了 scalar 接口和 pyspark 接口，同理 pip/conda 安装的 pyspark 包也整合了 spark，择一即可。

打开 `~/.bashrc` 文件，按如下配置环境变量。

```shell
# Spark environment variables
export SPARK_HOME=/usr/local/spark2.4.1-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
# Spark support for python  # 获取所有 zip 文件路径   # 修改间隔符为: # 输出所有 zip 文件  # 要在最后加否则前面出错
export PYTHONPATH=$(ZIPS=("$SPARK_HOME"/python/lib/*.zip); IFS=:; echo "${ZIPS[*]}"):$PYTHONPATH
export PYSPARK_PYTHON=python3
# Hadoop classpath for "Hadoop free" spark version  # 这里直接执行 hadoop 命令获取 classpath
export SPARK_DIST_CLASSPATH=$(hadoop classpath)
```

修改保存后，更新环境变量，并进行测试。

```shell
$ source ~/.bashrc
$ spark-shell # 测试 scalar 接口
$ pyspark     # 测试 python 接口
```

## 2 

## 数据结构

### PyArrow

### Parquet

教程：https://juejin.cn/post/6844903462572916743

读写：https://arrow.apache.org/docs/python/parquet.html#finer-grained-reading-and-writing
