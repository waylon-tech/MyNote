## 目录

[toc]

#### 环境迁移

**在线迁移**

1. 导出 `environment.yml` 文件

   ```shell
   $ conda env export > environment.yml
   ```

2. 在 `environment.yml` 文件路径下使用

   ```shell
   $ conda env create -f environment.yml
   ```
   
   移植过来的环境只包括用 `conda install` 命令安装的包。

3. 导出 `requirement.txt` 文件

   ```shell
   $ pip freeze > requirements.txt
   ```

4. 安装 `requirement.txt` 内的库

   ```shell
   $ pip install -r requirements.txt
   ```

**离线迁移**

1. 安装 `conda-pack`

   ```shell
   $ pip install conda-pack
   ```

2. 从服务器 A 打包 `conda` 环境

   ```shell
   $ conda pack -n wxs  //wxs 为环境名，打包之后的文件名为 wxs.tar.gz
   ```

3. 解压环境到服务器 B 中的 `conda` 环境目录 `xxx/anaconda3/envs/`

   ```shell
   $ mkdir -p wxs
   $ tar -xzf wxs.tar.gz -C wxs
   ```

4. 激活使用

   ```shell
   $ conda activate wxs
   ```
