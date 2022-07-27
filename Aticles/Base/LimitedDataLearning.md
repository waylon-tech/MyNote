

## 目录

[toc]

## 2 数据增强

### 2.1 数据增强概述

**(1) 数据增强的出发点**

我们日常生活中能够获得的数据往往非常有限，这里的**核心问题**是：如何用有限的数据训练出好的神经网络模型呢？

首先解答一个**基本问题**作为切入：为什么需要大量的数据？

因为当今是大模型时代，如下图所示，主流的模型都使用到了数兆的数据。

![LimitedDataLearning-DataAugmentation-1](img\LimitedDataLearning-DataAugmentation-1.png)

大模型往往意味着巨大的参数数量，因此就需要大量的数据。

**(2) 数据增强的思想**

数据增加的思想是，通过对现有数据集进行微小的改动，来获得更多的数据。

但使用数据增强有一个前提，那就是所要训练的模型必须对数据的改动具有**稳定性**，然后才可以**综合使用**各种方法来修改数据。

**(3) 数据增强的作用与局限**

数据增强能够增加数据集中相关数据的数量，减少不相关特征在数据集中出现的频率，从而帮助模型抓住重要特征。

但要谨记的是，神经网络的表现永远取决于数据中所蕴含的信息的质与量。

### 2.2 数据增强的方法

#### 2.2.1 数据增强的时机

在机器学习过程（管线）中，什么时候进行数据增强呢？

首先，增强过程肯定在喂数据给模型之前。然后，在这里有两种选择：

* 离线增强
  * 在执行一定数量的增强操作后停止操作
  * 适用于小数据集
* 在线增强
  * 每次增强小批量的数据，然后喂入模型
  * 适用于大数据集

#### 2.2.2 常用数据增强方法

叙述下面的方法时，先暂时忽略图像边界的处理。

**(1) 翻转**

可以水平或垂直翻转图像。数据增强因子 $=2\sim4$。

![LimitedDataLearning-DataAugmentation-2](img\LimitedDataLearning-DataAugmentation-2.png)

```python
# NumPy. 'img' = A single image
flip1 = np.fliplr(img)
```

**(2) 旋转**

图像的维度可能不再相同，要注意。数据增强因子$=2\sim4$。

![LimitedDataLearning-DataAugmentation-3](img\LimitedDataLearning-DataAugmentation-3.png)

```python
# PIL Image
from PIL import Image
#读取图像
im = Image.open("lenna.jpg")
im.show()
# 指定逆时针旋转的角度
im_rotate = im.rotate(45) 
im_rotate.show()
```

**(3) 放缩**

图像可以向外或向内放缩。数据增强因子$=\text{任意}$。

![LimitedDataLearning-DataAugmentation-4](img\LimitedDataLearning-DataAugmentation-4.png)

```python
# Scikit Image.
# 	'img' = Input Image
# 	'scale' = Scale factor
# For details about 'mode', checkout the interpolation section below.
# Don't forget to crop the images back to the original size (for # scale_out) 
scale_out = skimage.transform.rescale(img, scale=2.0, mode='constant')
scale_in = skimage.transform.rescale(img, scale=0.5, mode='constant')

# PIL Image
from PIL import Image
#读取图像
im = Image.open("lenna.jpg")
im.show() 
#原图像缩放为128x128
im_resized = im.resize((128, 128))
im_resized.show()
```

**(4) 裁剪**

随机选取一块区域，放大为原始图像的大小。数据增强因子$=\text{任意}$。

![LimitedDataLearning-DataAugmentation-5](img\LimitedDataLearning-DataAugmentation-5.png)

```python
# PIL Image
from PIL import Image
# 打开图像文件，注意是当前路径，比如这个py文件在桌面存放，那图片也放桌面
im = Image.open('连连看.png')
#看看图片的宽和高
w, h = im.size
print(w,h)
#可以用画图工具看需要裁剪的位置
#crop（x1，y1，x2，y2） 裁剪的是矩形  左上（x1，y1）  到右下（x2，y2）
cropim = im.crop((100, 1150, 1000, 1900)) 
#保存，也是保存在当前目录
cropim.save("cropim.png")
```

**(5) 平移**

可以在 `x` 轴或 `y` 轴甚至任意方向上平移图像。数据增强因子$=\text{任意}$。

![LimitedDataLearning-DataAugmentation-6](img\LimitedDataLearning-DataAugmentation-6.png)

```python

```

**(6) 高斯加噪**

当作用不大的特征出现频率过高时，所学习的神经网络模型往往出现过拟合现象。向数据中添加高斯噪声，能够有效降低高频特征。数据增强因子$=2\times$。

除了高斯噪声，还可以添加一些非空白噪声：椒盐噪声，也称为脉冲噪声，它是一种随机出现的白点或者黑点。它与高斯噪声类似，但是有更高的扰动水平。

![LimitedDataLearning-DataAugmentation-7（左高斯右椒盐）](img\LimitedDataLearning-DataAugmentation-7.png)

```python
# skimage Image
import skimage
origin = skimage.io.imread("./lena.png")
noisy = skimage.util.random_noise(origin, mode='gaussian', var=0.01)
```

#### 2.2.3 高级数据增强方法

在真实世界中，自然数据仍然会存在各种难以通过上述简单方法规避的特征。例如：数据中的单一风格（季节、天气）。

* 一种减轻这种状况的方法是获取风格**更加多样的数据**，但这往往非常困难。

* 另一种开创性的方法是**条件 GAN**，它能将一张图片的风格迁移到另一张图像上。

* 最后，GAN 虽然很有效，但是计算量也大，于是可以考虑使用**神经风格迁移**（Neural Style Transfer）技术，它是更早期的方法。

### 2.3 插补方法

图像进行平移、缩放等操作后，会留下空白区域，有以下填充方法。

**(1) 常数插值**

在空白区域填充常数值，适合单色背景的图片。

**(2) 边缘插值**

垂直延伸边界像素点的值至图片框，适合轻微的平移操作。

**(3) 反射插值**

以边界为轴，填充图片的镜像（包括边界），适合连续或自然的背景（如树、山）。

**(4) 对称插值**

以边界为轴，填充图片的进行（不包括边界），与反射插值类似。

**(5) 包裹插值**

重新平铺图像，该方法不适合大部分情景。

最后，可以根据自己的实际需要，设计、组合相应合适的数据增加方法。