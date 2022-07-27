## 目录

[toc]

## 1 PyTorch

PyTorch 是一个开源的机器学习框架。

### 1.1 `torch`

`torch` 包（package）含多维度 tensors 的数据结构及其相关操作。除此之外，还提供这些 tensors 的高效序列化工具等。

#### 1.1.1 Tensors 相关操作

**操作函数**

* `torch.numel(input) → int` - 返回 `input` 的元素总数

  * `input (Tensor)` - 输入 tensor

  ```python
  >>> a = torch.randn(1, 2, 3, 4, 5)
  >>> torch.numel(a)
  120
  >>> a = torch.zeros(4,4)
  >>> torch.numel(a)
  16
  ```

**创建函数**

* `torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)`

  返回全一个 `0` 的 tensor，等价于 `torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.

  * `input (Tensor)` - `input` 的大小决定返回值的大小
  * `dtype (torch.dtype, optional)` - tensor 的数据类型
  * `layout (torch.layout, optional)` - tensor 的布局？
  * `device (torch.device, optional)` - tensor 的使用设备
  * `requires_grad (bool, optional)` - autograd 是否记录返回值上的操作（反向传播求导使用）
  * `memory_format (torch.memory_format, optional) ` - tensor 的记忆模式？
  * 返回值：`tensor` 类型
  
  ```python
  >>> input = torch.empty(2, 3)
  >>> torch.zeros_like(input)
  tensor([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.]])
  ```

* `torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)`

  返回全一个 `1` 的 tensor，参数和 `torch.zeros_like(...)` 一样。

**形状函数**

* `torch.chunk(input, chunks, dim=0) → List of Tensors` - 将一个 tensor 拆分为指定数量的 chunks，每个都是输入 tensor 的 view
  * `input (Tensor)` - tensor 拆分
  * `chunks (int)` - 返回的 chunks 数量
  * `dim (int)` - 拆分的维度

* `torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor` - 沿 `dim` 指定的轴聚集值

  聚集：在指定的轴上，根据 `index` 指定的下标，选择元素重组成一个新的 tensor

  * `input (Tensor)` - 源向量
  * `dim (int)` - 要索引的维度
  * `index (LongTensor)` - 要聚集的元素索引
  * `sparse_grad (bool, optional)` - `True` 时表示 `input` 的梯度是一个 sparse tensor
  * `out (Tensor, optional)` - 目标 tensor

  ```python
  >>> t = torch.tensor([[1, 2], [3, 4]])
  >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
  tensor([[ 1,  1],
          [ 4,  3]])
  ```

* `torch.masked_select(input, mask, *, out=None) → Tensor` - 根据 `mask` 选择 `input` 中的元素并返回 1-D tensor

  * `input (Tensor)` - 输入 tensor
  * `mask (BoolTensor)` - 用于选择的 tensor

  ```python
  >>> x = torch.randn(3, 4)
  >>> x
  tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
          [-1.2035,  1.2252,  0.5002,  0.6248],
          [ 0.1307, -2.0608,  0.1244,  2.0139]])
  >>> mask = x.ge(0.5)
  >>> mask
  tensor([[False, False, False, False],
          [False, True, True, True],
          [False, False, False, True]])
  >>> torch.masked_select(x, mask)
  tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
  ```

* `torch.t(input) → Tensor` - 对二维 tensor 进行转置

* `torch.squeeze(input, dim=None, *, out=None) → Tensor` - 移除 tensor 中维度为 1 的

* `torch.unsqueeze(input, dim) → Tensor` - 返回一个新的 tensor，其在指定位置插入一个大小为 `1` 的维度

  * `input (Tensor)` - 输入 tensor
  * `dim (int)` - 插入单个维度的索引位置，取值范围在 `[-input.dim() - 1, input.dim() + 1)`，负值将从 `input.dim() + 1` 倒着算位置
  * 返回值 - tensor，其与 `input` 共享底层数据

  ```python
  >>> x = torch.tensor([1, 2, 3, 4])
  >>> torch.unsqueeze(x, 0)
  tensor([[ 1,  2,  3,  4]])
  >>> torch.unsqueeze(x, 1)
  tensor([[ 1],
          [ 2],
          [ 3],
          [ 4]])
  ```

### 1.2 `torch.nn`

`torch.nn` 是 graph 的基本构造块，对深度学习的各种网络层进行了抽象。

* 类 `torch.nn.parameter.Parameter` - `torch.Tensor` 的子类

  理解：可以看做是一个类型转换函数，将一个不可训练的 `Tensor` 类型转换成可以训练的 `parameter` 类型并绑定到 `torch.nn.Module` 里面。这样，该 tensor 就成为模型的一部分，可以用 `torch.nn.Module.parameters()` 迭代出来，也就可以训练和更新。

  区别：`torch.Tensor(requires_grad=True)` 只将参数变成可训练的，并没有绑定到 `module` 的参数列表里面。

#### 1.2.1 容器

* `torch.nn.Module` - PyTorch 中所有神经网络 modules 的 基类，其可以作为属性进行嵌套从而实现树形的神经网络结构

  属性：

  * `training (bool)` - 表示该 module 在哪个模式（训练 or 验证）

  方法：

  * `add_module(name, module)` - 在当前 module 添加一个子 module
    * `name (string)` - 子 module 的名称，可以通过该名称访
    * `module (Module)` - 要添加的子 module

  * `apply(fn)` - 迭代每个子 module 和 self，来传入到 `fn` 函数中，一般用于初始化各个 module 的参数

    * `fn (Module -> None)` - 应用的每个 submodule 的函数
    * 返回值 - self

    ```python
    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    ```
    
  * `train(mode=True)` - 将当前 module 设置为 training 模式

    * `mode (bool)` - `True` 为 training 模式，`False` 为 evaluation 模式

    【注】这个方法只适用于某些 modules，详见具体的 modules。

  * `eval()` - 将当前 module 设置为 evaluation 模式，等价于 `self.train(False)`

    * 返回值 - `self`

    【注】这个方法只适用于某些 modules，详见具体的 modules。

  * `named_parameters(prefix='', recurse=True)` - 返回 module 参数的可迭代对象，包括参数名和参数本身

    * `prefix (str)` - 增加的所有参数名前面的前缀
    * `recurse (bool)` - 是否涉及子 modules
    * 返回值 `(string, Parameter)` - 包含名字和参数的二元组

    ```python
    >>> for name, param in self.named_parameters():
    >>>    if name in ['bias']:
    >>>        print(param.size())
    ```

  * `parameters(recurse=True)` - 返回 module 参数的可迭代对象

    * `recurse (bool)` - 是否涉及子 modules
    * 返回值 `Parameter` - module 参数
    
    ```python
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
    ```

* `torch.nn.Sequential(*args)` - 顺序容器，模型将按顺序添加到其中，输入也会自动有序在其中传播得到输出

  ```python
  # Example of using Sequential
  model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
          )
  
  # Example of using Sequential with OrderedDict
  model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1,20,5)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5)),
            ('relu2', nn.ReLU())
          ]))
  ```

* `torch.nn.ModuleList(modules=None)` - 在列表中保存 submodules，只是保存，不会像 `torch.nn.Sequential` 一样运行，可以取出来操作

  * `modules (iterable, optional)` - 要添加的可迭代 modules

  ```python
  class MyModule(nn.Module):
      def __init__(self):
          super(MyModule, self).__init__()
          self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
  
      def forward(self, x):
          # ModuleList can act as an iterable, or be indexed using ints
          for i, l in enumerate(self.linears):
              x = self.linears[i // 2](x) + l(x)
          return x
  ```

  * `append(module)` - 将给定的 module 添加到 `ModuleList` 对象末尾
    * `module (nn.Module)` - 要添加的 module
  * `extend(modules)` - 从一个 python 可迭代对象中将 module 添加到 `ModuleList` 对象末尾
    * `modules (iterable)` - modules 的可迭代对象
  * `insert(index, module)` - 在给定位置前插入给定的 module
    * `index (int)` - 插入的位置
    * `module (nn.Module)` - 要插入的 module

#### 1.2.2 卷积层

* `torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)` - 二维卷积层

  一般来说，输入大小 $(N,C_{\text{in}},H_{\text{in}},W_{\text{in}})$ 和输出大小 $(N,C_{\text{out}},H_{\text{out}},W_{\text{out}})$ 直接的关系为：
  $$
  \text{out}(N_i,C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j})+\sum_{k=0}^{C_{\text{in}}-1} \text{weight}(C_{\text{out}},k) \star \text{input}(N_i,k)
  $$
  其中 $\star$ 为二维互相关操作（现在意义下的卷积），$N$ 为批大小，$C$ 为通道数，$H$ 为平面高度，$W$ 为平面宽度。

  * `in_channels (int)` - 输入通道数
  * `out_channels (int)` - 输出通道数
  * `kernel_size (int or 2_int_tuple)` – 卷积核大小
  * `stride (int or 2_int_tuple, optional) `– 卷积步长，默认：`1`
  * `padding (int, 2_int_tuple or str, optional)` – `{‘valid’, ‘same’}` 或 `(int)`，表示四边进行 padding 的大小，默认：`0`
  * `padding_mode (string, optional)` – `'zeros'`, `'reflect'`, `'replicate'` 或 `'circular'`，默认：`'zeros'`
  * `dilation (int or 2_int_tuple, optional)` – 卷积核元素之间的间距，用于空洞卷积，默认：`1`
  * `groups (int, optional)` – 分块卷积，每个卷积核只能看到部分的通道，`in_channels` 和 `out_channels` 必须能整除它，默认：`1`
  * `bias (bool, optional)` – 为 `True`，添加一个可学习的偏倚到输出中，默认：`True`

  形状：[CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

  变量：[CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

  ```python
  >>> # With square kernels and equal stride
  >>> m = nn.Conv2d(16, 33, 3, stride=2)
  >>> # non-square kernels and unequal stride and with padding
  >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
  >>> # non-square kernels and unequal stride and with padding and dilation
  >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
  >>> input = torch.randn(20, 16, 50, 100)
  >>> output = m(input)
  ```

#### 1.2.3 池化层

* `torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)` - 二维最大池化层

#### 1.2.4 padding 层

* `torch.nn.ZeroPad2d(padding)` - 二维零值 padding 层

#### 1.2.5 非线性激活层

* `torch.nn.MultiheadAttention(...)` - 多头注意力机制

  计算公式为：
  $$
  \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
  \\
  \text{head}_1 = \text{Attention}(QW^Q_i,KW^K_i,VW^V_i)
  $$
  其中投影矩阵为 $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 和 $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$. Transformer 一文中，取 $h=8$，$d_k=d_v=d_{\text{model}}/h=64$，由于每个头维度减小，故总的花费与满维度的 single-head attention 近似。

  详细原理参看 Articles/NLP.md/Transformer.

  * `embed_dim` – 模型总维度，即 $d_{\text{model}}$
  * `num_heads` – 注意力头个数，即 $h$
  * `dropout` – 在 attn_output_weights 上的一个 dropout 层，默认值：`0.0`
  * `bias` – 添加 bias 作为模块参数，默认值：`True`
  * `add_bias_kv` – 在 key 和 value 序列中的 `dim=0` 处添加 bias 
  * `add_zero_attn` – 在 key 和 value 序列中的 `dim=1` 处添加一个值为 `0` 的 batch
  * `kdim` – key 特征维度，即 $d_k$
  * `vdim` – value 特征维度，即 $d_v$
  * 返回值 `attn_output` - 经 attention 加权后的 value，维度为 $(\text{num}\_\text{query}, \text{batch}\_\text{size}, d_{\text{model}})$​​​=$(L,N,E)$​​​
  * 返回值 `attn_output_weights` - 经 attention 计算出来的权重矩阵，维度为 $(\text{batch}\_\text{size},\text{num}\_\text{query},\text{num}\_\text{key})$=$(N,L,S)$
  * 【注】如果 `kdim` 和 `vdim` 为 `None`，它们将被设置为 `embed_dim`

  ```python
  >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
  >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
  ```
  * `forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None)` - attention 计算函数

    * `key, value (query,)` – 映射一个 query 到一个 key-value 对集合到输出上

      * `query` 的维度为 $(L,N,E)$，即 target sequence length，batch size，embedding dimension
      * `key` 和 `value` 的维度为 $(S,N,E)$，即 source sequence length，batch size，embedding dimension

    * `key_padding_mask` – 每一个 batch 的句子长度不可能完全相同，会使用 padding 进行填充，而这里的参数是用来“遮挡”这些 padding 的

      * mask 取值 `0` 或 `1`，维度为 $(N,S)$，即 batch size，source sequence length
      * 先取得 key 中有 padding 的位置，然后把 mask 里相应位置设置为 1，这样 attention 就会把 key 相应的部分变为 `-inf`（见下图）

      ![在这里插入图片描述](img/20200617175743395.png)

    * `need_weights` – 输出 `attn_output_weights`

    * `attn_mask` – 2D 或 3D mask，用于遮挡某些位置的 attention 

      * 2D mask 将会广播到所有的 batch，维度为 $(L,S)$，即 target sequence length，source sequence length
      * 3D mask 允许为每个 batch 的输入指定不同的 mask，维度为 $(N⋅\text{num}\_\text{heads},L,S)$​，多了 batch size 乘 `num_heads`

* `torch.nn.ReLU(inplace=False)` - 在每一个元素上应用校正的线性单元函数

  RELU 函数公式为：
  $$
  ReLU(x) = (x)^+ = \max{0,x}
  $$
  <img src="img/ReLU.png" alt="../_images/ReLU.png" style="zoom: 67%;" />

  * `inplace` - 表示是否直接在输入上进行处理

  形状：

  * `input` - $(N,*)$，其中 $*$ 表示其他维度
  * `output` - $(N,*)$，同上

  ```python
  >>> m = nn.ReLU()
  >>> input = torch.randn(2)
  >>> output = m(input)
  ```

* `torch.nn.Tanh` - 在每一个元素上应用 $\tanh$ 函数

  Tanh 函数公式为：
  $$
  Tanh(x) = \tanh(x) = \frac{\exp{(x)}-\exp{(-x)}}{\exp{(x)}+\exp{(-x)}}
  $$
  <img src="img/Tanh.png" alt="../_images/Tanh.png" style="zoom:67%;" />

  形状：

  * `input` - $(N,*)$，其中 $*$ 表示其他维度
  * `output` - $(N,*)$，同上

  ```python
  >>> m = nn.Tanh()
  >>> input = torch.randn(2)
  >>> output = m(input)
  ```

#### 1.2.5 正则化层

* `torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)`

  二维正则化层

  在一个标准的 4D input（mini-batch, width, height, hidden dim = $(N,H,W,C)$） 上应用批正则化，公式为：
  $$
  y = \frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}*\gamma+\beta
  $$
  参考文献： [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

  $\mathrm{E}[x]$ 和 $\mathrm{Var}[x]$（同 `torch.var(input, unbiased=False)`）在每个 mini-batch 维度上计算；

  $\gamma$（默认全1） 和 $\beta$（默认全0） 是可学习参数向量（长度 $C$）；

  * `num_features` - 通道数 $C$

  * `eps` - 加到分母的那个 $\epsilon$，用于数值稳定性，默认：`1e-5`

  * `momentum` - running_mean 和 running_std 计算时的更新大小，可为 `None`，默认：`0.1`

    更新公式为：$\hat{x}_{\text{new}} = (1-\text{momentum})\times\hat{x}+\text{momentum}\times x_t$，$x$ 是待估计的统计量，$x_t$ 是新的观测值

  * `affline` - a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`
  * `track_running_stats` - 见 `1.8` 笔记

  形状：

  * 输入：$(N,C,H,W)$
  * 输出：$(N,C,H,W)$

#### 1.2.9 线性层

* `torch.nn.Linear(in_features, out_features, bias=True)` - 线性变换层

  线性变换层的数学公式表达为：$y = x A^T + b$.

  * `in_features` - 每个输入样本的大小
  * `out_features` - 每个输出样本的大小
  * `bias` - `False` 时该层不会学习偏倚项 $b$ 的大小

  形状：

  * `input` - $(N,*,H_{in})$​，其中 $*$​ 表示额外维度，$H_{in} = \text{in}\_\text{features}$​
  * `output` - $(N,*,H_{out})$​，其中除了 $H_{out}=\text{out}\_\text{features}$​ 外，其他与 `input` 一样

  变量：

  * `~Linear.weight` - 形状为 $(\text{in}\_\text{features},\text{out}\_\text{features})$ 的可学习权重矩阵，初始化为 $\mathcal{\mu}(-\sqrt{k},\sqrt{k})$，$k=\frac{1}{\text{in}\_\text{features}}$
  * `~Linear.bias` - 形状为 $(\text{out}\_\text{features})$ 的可学习偏移向量，如果 `bias = True`， 初始化为 $\mathcal{\mu}(-\sqrt{k},\sqrt{k})$，$k=\frac{1}{\text{in}\_\text{features}}$

  ```python
  >>> m = nn.Linear(20, 30)
  >>> input = torch.randn(128, 20)
  >>> output = m(input)
  >>> print(output.size())
  torch.Size([128, 30])
  ```

#### 1.2.12 sparse 层

* `torch.nn.Embedding(...)` - 一个简单的查找表，用于存储固定字典和大小的嵌入

  该网络常用于存储 embeddings，可用下标（列表）提取它们，它的**映射关系就是 `token_id -> token_vector`**，并且支持矩阵处理。

  * `num_embeddings (int)` - embeddings 字典的大小
  * `embedding_dim (int)` - 每个 embedding 向量的大小
  * `padding_idx (int, optional)` - 指定后，`padding_idx` 下的输入不会计算梯度，并且在处理序列时， `embedding` 会**将填充位置的值映射到它**
  * `max_norm (float, optional)` - 指定后，每个范数大于它的 embedding 向量将会重新正则化
  * `norm_type (float, optional)` - 范数类型，表示 p 范数的 p 值
  * `scale_grad_by_freq (boolean, optional)` - 给定后，将会用 mini-batch 中的逆文档频率调整（scale）梯度
  * `sparse (bool, optional)` - `True` 表示梯度（即权重矩阵）是稀疏的

  变量：

  * `~Embedding.weight (Tensor)` - 网络的可学习权重，形状为 `(num_embeddings, embedding_dim)`，用 $\mathcal{N}(0,1)$ 初始化

  形状：

  * `input` - `(*)`
  * `output` - `(*, H)`，其中 `H=embedding_dim`
  
  【注1】只有少数优化器支持 sparse gradients：`optim.SGD`（CUDA 和 CPU），`optim.SparseAdam`（CUDA 和 CPU）。
  
  ```python
  >>> # an Embedding module containing 10 tensors of size 3
  >>> embedding = nn.Embedding(10, 3)
  >>> # a batch of 2 samples of 4 indices each
  >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
  >>> embedding(input)
  tensor([[[-0.0251, -1.6902,  0.7172],
           [-0.6431,  0.0748,  0.6969],
           [ 1.4970,  1.3448, -0.9685],
           [-0.3677, -2.7265, -0.1685]],
  
          [[ 1.4970,  1.3448, -0.9685],
           [ 0.4362, -0.4004,  0.9400],
           [-0.6431,  0.0748,  0.6969],
           [ 0.9124, -2.3616,  1.1151]]])
  
  
  >>> # example with padding_idx
  >>> embedding = nn.Embedding(10, 3, padding_idx=0)
  >>> input = torch.LongTensor([[0,2,0,5]])
  >>> embedding(input)
  tensor([[[ 0.0000,  0.0000,  0.0000],
           [ 0.1535, -2.0309,  0.9315],
           [ 0.0000,  0.0000,  0.0000],
           [-0.1655,  0.9897,  0.0635]]])
  
  >>> # example of changing `pad` vector
  >>> padding_idx = 0
  >>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
  >>> embedding.weight
  Parameter containing:
  tensor([[ 0.0000,  0.0000,  0.0000],
          [-0.7895, -0.7089, -0.0364],
          [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
  >>> with torch.no_grad():
  ...     embedding.weight[padding_idx] = torch.ones(3)
  >>> embedding.weight
  Parameter containing:
  tensor([[ 1.0000,  1.0000,  1.0000],
          [-0.7895, -0.7089, -0.0364],
          [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
  ```
  
  * `from_pretrained(...)` - 类方法，给定 2 维 tensor 创建 embedding 层
  
    * `embeddings (Tensor)` - 包含 embeddings 权重的 `FloatTensor`，维度为 `(num_embeddings, embedding_dim)`
    * `freeze (boolean, optional)` - `True` 时，权重不会在训练是更新，等价于 `embedding.weight.requires_grad = False`，默认 `True`
    * `padding_idx (int, optional)` - 指定时，在 `padding_idx` 的输入不会贡献梯度，对应 embedding 向量训练时不更新
    * `max_norm (float, optional)` - 同上
    * `norm_type (float, optional)` - 同上
    * `scale_grad_by_freq (boolean, optional)` - 同上
    * `sparse (bool, optional)` - 同上
  
    ```python
    >>> # FloatTensor containing pretrained weights
    >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
    >>> embedding = nn.Embedding.from_pretrained(weight)
    >>> # Get embeddings for index 1
    >>> input = torch.LongTensor([1])
    >>> embedding(input)
    tensor([[ 4.0000,  5.1000,  6.3000]])
    ```

#### 1.2.17 工具库

* `torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False)` - 裁剪可迭代参数的渐变范数

  BP 过程中会产生梯度消失/爆炸（偏导无限接近 `0`，导致长时记忆无法更新），简单粗暴的方法是设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值

  * `parameters (Iterable[Tensor] or Tensor)` - 一个由张量或单个张量组成的可迭代对象（模型参数），将梯度归一化
  * `max_norm (float or int)` - 梯度的最大范数
  * `norm_type (float or int)` - 所使用的范数类型。默认为L2范数，可以是无穷大范数（`'inf'`）
  * `error_if_nonfinite (bool)` - `True` 时，如果梯度参数中有 `nan`，`inf`，`-inf` 抛出异常
  * 返回值 - 参数的总范数(作为单个向量来看)

* `nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)` - 用 padding 值填充可变长度 Tensors
  
  * `sequence` - `list[Tensor]`，可变长度序列的列表
  * `batch_first` - `bool`，`True` 时的输出为 $B \times T \times *$，否则为 $T \times B \times *$
    * $T$ - 最长的序列长度
  * `padding_value` - `float`，padding 填充值
  
  ```python
  >>> from torch.nn.utils.rnn import pad_sequence
  >>> a = torch.ones(25, 300)
  >>> b = torch.ones(22, 300)
  >>> c = torch.ones(15, 300)
  >>> pad_sequence([a, b, c]).size()
  torch.Size([25, 3, 300])
  ```

### 1.3 `torch.nn.functional`

`torch.nn.functional` 包含深度学习众多的函数，它是上一个 `torch.nn` 对层的函数化形式。

#### 1.3.1 卷积函数

#### 1.3.2 池化函数

#### 1.3.3 非线性激活函数

* `torch.nn.functional.relu(input, inplace=False) → Tensor` - 在每一个元素上应用校正的线性单元函数

  它是 `torch.nn.ReLU(inplace=False)` 层的函数化形式，详见 1.2.5.

* `torch.nn.functional.log_softmax(input, dim=None, _stacklevel=3, dtype=None)` - 对数 softmax

  在数学上等价于 `log(softmax(x))`。

  * `input (Tensor)` - 输入
  * `dim (int)` - 计算的维度
  * `dtype (torch.dtype, optional)` - 返回 tensor 期望的数据类型

#### 1.3.4 线性函数

* `torch.nn.functional.linear(input, weight, bias=None)` - 对输入进行线性变换
  $$
  y = xA^T+b
  $$
  形状：

  * `input` - $(N, *, in\_features)$，其中 $N$ 为 batch_size，$*$ 为任意维度
  * `weight` - $(out\_features, in\_features)$
  * `bias` - $(out\_features)$
  * `output` - $(N, *, out\_features)$

#### 1.3.5 dropout 函数

#### 1.3.5 sparse 函数

#### 1.3.6 距离函数

#### 1.3.7 损失函数

* `torch.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')`

  交叉熵损失函数，综合 `log_softmax` 和 `nll_loss` 两种损失。

  * `input (Tensor)` - 输入，形状为 $(N,C)$，其中 $C$ 为类别数或二维损失 $(N,C,H,W)$ 或多维损失 $(N,C,d_1,d_2,\dots,d_K)$
  * `target (Tensor)` - 真值，形状为 $(N)$，其中每个值 $0 \leq \text{targets}[i] \leq C-1$
  * `weight (Tensor, optional)` - 对每个类别人工重新加权，必须是个 $C$ 维向量
  * `size_average (bool, optional)` - 已废弃，见 `reduction` 参数
  * `ignore_index (int, optional)` - 
  * `reduce (bool, optional)` - 已废弃，见 `reduction` 参数
  * `reduction (string, optional)` - 输出的降维方式，有下列选项
    * `'none'` - 不进行降维
    * `'batchmean'` - 输出求和并处以 `batch_size`
    * `'sum'` - 输出求和
    * `'mean'` - 输出求和处以元素个数，默认

* `torch.nn.functional.kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False)` - KL 散度

  参考资料：[Kullback-Leibler divergence Loss](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence)

  * `input` - 任意形状的 tensor
  * `target` - 形状和 `input` 一样
  * `size_average (bool, optional)` - 已废弃，见 `reduction` 参数
  * `reduce (bool, optional)` - 已废弃，见 `reduction` 参数
  * `reduction (string, optional)` - 输出的降维方式，有下列选项
    * `'none'` - 不进行降维
    * `'batchmean'` - 输出求和并处以 `batch_size`
    * `'sum'` - 输出求和
    * `'mean'` - 输出求和处以元素个数，默认
  * `log_target (bool)` - 表示 `target` 是否也是在对数空间上，建议传入对数空间上的 `target`，避免数值问题

#### 1.3.8 图像函数

* `torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)` - 按给定 `size` 或 `scale_factor` 进行上/下采样的插值函数

  <img src="img/1446032-20190823144053396-1331930578.png" alt="img" style="zoom:50%;" />

  * `input (Tensor)` - 输入 tensor

  * `size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int])` - 输出空间大小

  * `scale_factor (float or Tuple[float])` - 空间大小乘子，`tuple` 类型与 `size` 的维度匹配

  * `mode (str)` - 上采用的模式，可选 `'nearest'`（默认） | `'linear'`（3D） | `'bilinear'` | `'bicubic'`（4D） | `'trilinear'`（5D） | `'area'`

  * `align_corners (bool, optional)` - 像素角落对齐

    <img src="img/1446032-20190823152954205-86765483.png" alt="img" style="zoom:50%;" />

  * `recompute_scale_factor (bool, optional)`

* `torch.nn.functional.pad(input, pad, mode='constant', value=0)` - padding 函数

  输入中各个维度的 padding 大小从最后一维开始向前匹配，padding 最后一维时 `pad = (padding_top, padding_bottom)`，padding 最后两维时 `pad = (padding_left, padding_right, padding_top, padding_bottom)`，以此类推。

  * `input (Tensor)` – $N$ 维张量
  * `pad (tuple)` - $m$ 个元素的 `tuple`，其中 $m$ 为偶数且 $\frac{m}{2} \le$​ 输入维度​ 
  * `mode` - padding 模式，其中可选 `'constant'`, `'reflect'`, `'replicate'` or `'circular'`，默认 `'constant'`
  * `value` - `mode='constant'` 模式下的 padding 填充值，默认 `0`

  ```c++
  >>> t4d = torch.empty(3, 3, 4, 2)
  >>> p1d = (1, 1) # pad last dim by 1 on each side
  >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
  >>> print(out.size())
  torch.Size([3, 3, 4, 4])
  >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
  >>> out = F.pad(t4d, p2d, "constant", 0)
  >>> print(out.size())
  torch.Size([3, 3, 8, 4])
  >>> t4d = torch.empty(3, 3, 4, 2)
  >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
  >>> out = F.pad(t4d, p3d, "constant", 0)
  >>> print(out.size())
  torch.Size([3, 9, 7, 3])
  ```

### 1.4 `torch.Tensor`

PyTorch 定义了 10 中类型的 tensor，并且分为 GPU 和 CPU 设备。

#### 1.4.1 Tensor 对象方法

`torch.Tensor` 的方法既有全局方法，又有对象方法。这里主要做一个归纳，全局方法将引用到 `1.1.1 Tensors`，对象方法这里详解。

* `Tensor.chunk(chunks, dim=0) → List of Tensors`

  将一个 tensor 拆分为指定数量的 chunks，每个都是输入 tensor 的 view，源自 `torch.chunk(input, chunks, dim=0) → List of Tensors`

* `Tensor.copy_(src, non_blocking=False) → Tensor` - 从 `scr` 中复制元素到自身然后返回

  * `src (Tensor)` - 要复制的源 tensor
  * `non_blocking (bool)` - 为 `True` 时且此拷贝位于 CPU 和 GPU 之间，则允许异步拷贝；对于其他情况，该参数没有任何效果。

  【注】`src` 对于调用者来说必须是可广播的（broadcastable）

* `Tensor.element_size() → int` - 返回单个元素的 bytes 大小

  ```python
  >>> torch.tensor([]).element_size()
  4
  >>> torch.tensor([], dtype=torch.uint8).element_size()
  1
  ```

* `Tensor.expand(*sizes) → Tensor` - 扩展当前 tensor 为指定的 size

  * `*sizes (torch.Size or int...)` - 指定的 size

  ```python
  >>> x = torch.tensor([[1], [2], [3]])
  >>> x.size()
  torch.Size([3, 1])
  >>> x.expand(3, 4)
  tensor([[ 1,  1,  1,  1],
          [ 2,  2,  2,  2],
          [ 3,  3,  3,  3]])
  >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
  tensor([[ 1,  1,  1,  1],
          [ 2,  2,  2,  2],
          [ 3,  3,  3,  3]])
  ```

* `Tensor.expand_as(other) → Tensor` - 扩展当前 tensor 的 size 与 `other` 一样

* `Tensor.fill_(value) → Tensor` - 用特定值填充自身

* `Tensor.masked_fill(mask, value) → Tensor` - 基于 mask 进行填充，`Tensor.masked_fill_()` 的外部版本

* `Tensor.masked_fill_(mask, value)` - 基于 mask 进行填充，有 `mask` 为 `True` 的地方填充为 `value` 值

  * `mask (BoolTensor)` - 布尔遮罩
  * `value (float)` - 填充值

* `Tensor.masked_select(mask) → Tensor` - 基于 mask 进行选择，源自 `torch.masked_select(input, mask, *, out=None)`

* `Tensor.new() → Tensor` - 创建一个新的 Tensor，其 `type` 和 `device` 都和原有 Tensor 一致，且无内容

  ```python
  inputs = torch.randn(m, n)
  new_inputs = inputs.new()  				# 对象方法
  new_inputs = torch.Tensor.new(inputs)  	# 全局方法
  ```

* `Tensor.numel() → int` - 返回元素总数，源自 `torch.numel(input) → int`

* `Tensor.nelement() → int` - `Tensor.numel()` 的别名，返回元素总数

* `Tensor.normal_(mean=0, std=1, *, generator=None) → Tensor` - 将自身填充为 $N(\text{mean},\text{std}^2)$ 的正态分布值

* `Tensor.record_stream(stream)` - 保证 tensor 内存不被另一个 tensor 重用，直到当前流队列中的工作结束

* `Tensor.repeat(*sizes) → Tensor` - 在特定的维度上重复当前的 tensor，不同于 `expand`，本函数会拷贝数据

  ```python
  >>> x = torch.tensor([1, 2, 3])
  >>> x.repeat(4, 2)
  tensor([[ 1,  2,  3,  1,  2,  3],
          [ 1,  2,  3,  1,  2,  3],
          [ 1,  2,  3,  1,  2,  3],
          [ 1,  2,  3,  1,  2,  3]])
  >>> x.repeat(4, 2, 1).size()
  torch.Size([4, 2, 3])
  ```

* `Tensor.scatter(dim, index, src) → Tensor` - `Tensor.scatter_()` 的外部版本

* `Tensor.scatter_(dim, index, src, reduce=None) → Tensor` - 将 `src` 中数据根据 `index` 中的索引按照 `dim` 的方向填进 `input` 中

  解析：里面的逻辑演示如下，抓住 **`index` 是对 `scr` 元素的位置映射，方向由 `dim` 决定**

  <img src="img/20200223172556360.png" alt="在这里插入图片描述" style="zoom:50%;" />

  ```python
  self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
  self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
  self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
  ```

  - `target` - 目标张量，即调用该函数的对象，将在该张量上进行映射
  - `dim (int)` - 指定轴方向，定义了填充方式。对于二维张量，`dim=0` 表示逐列进行行填充，而 `dim=1` 表示逐列进行行填充
  - `index (LongTensor)` - 按照轴方向，在`target`张量中需要填充的位置
  - `src (Tensor or float)` - 即源张量，将把该张量上的元素逐个映射到目标张量上
  - `reduce (str, optional)` - 可选 `'add'` 或 `'multiply'`

  ```python
  >>> src = torch.arange(1, 11).reshape((2, 5))
  >>> src
  tensor([[ 1,  2,  3,  4,  5],
          [ 6,  7,  8,  9, 10]])
  >>> index = torch.tensor([[0, 1, 2, 0]])
  >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
  tensor([[1, 0, 0, 4, 0],
          [0, 2, 0, 0, 0],
          [0, 0, 3, 0, 0]])
  >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
  >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
  tensor([[1, 2, 3, 0, 0],
          [6, 7, 0, 0, 8],
          [0, 0, 0, 0, 0]])
  
  >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
  ...            1.23, reduce='multiply')
  tensor([[2.0000, 2.0000, 2.4600, 2.0000],
          [2.0000, 2.0000, 2.0000, 2.4600]])
  >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
  ...            1.23, reduce='add')
  tensor([[2.0000, 2.0000, 3.2300, 2.0000],
          [2.0000, 2.0000, 2.0000, 3.2300]])
  ```

* `Tensor.t() → Tensor` - 对二维 tensor 进行转置，源自 `torch.t(input) → Tensor`

* `Tensor.squeeze(dim=None) → Tensor` - 移除 tensor 中维度为 1 的，源自 `torch.squeeze(input, dim=None, *, out=None) → Tensor`

* `Tensor.unsqueeze(dim) → Tensor` - 在指定位置添加一个维度然后返回，源自 `torch.unsqueeze(input, dim) → Tensor`

* `Tensor.zero_() → Tensor` - 将自身填充为全 `0` tensor


#### 1.4.2 Tensor 对象属性

每个 `torch.Tensor` 都有 `torch.dtype`，`torch.device`，`torch.layout` 属性。

1. **`torch.dtype`**

2. **`torch.device`**

`torch.device` 类的对象表示 `torch.Tensor` 被分配的设备，该对象包含一个设备类型（`cpu` 或 `cuda`）和该类型设备的编号。

构造方式：

```python
# string 方法
>>> torch.device('cuda:0')
device(type='cuda', index=0)

>>> torch.device('cpu')
device(type='cpu')

>>> torch.device('cuda')  # current cuda device
device(type='cuda')
```

```python
# 序号方法
>>> torch.device('cuda', 0)
device(type='cuda', index=0)

>>> torch.device('cpu', 0)
device(type='cpu', index=0)
```

【注】如果没有提供设备编号，则会使用当前设备类型的编号。

【技】需要 `torch.device` 类型的参数，可以直接使用字符串进行初始化：

```python
>>> cuda1 = torch.device('cuda:1')
>>> torch.randn((2,3), device=cuda1)
# 等价于
>>> torch.randn((2,3), device='cuda:1')
```

3. **`torch.layout`**

### 1.5 `torch.cuda`

这个包增加对 CUDA tensor 类型的支持，实现的函数与 CPU tensor 的一样，但是使用的是 GPU 计算。

* `torch.cuda.current_stream(device=None)` - 返回给定设备中当前所选的 `Stream`
  * `device (torch.device or int, optional)` - 所选的设备，如果为 `None` 则值从 `current_device()` 获得

* `torch.cuda.set_device(device)` - 设置当前设备

  * `device` - `torch.device` 或 `int` 类型

  【注】不推荐使用这个函数，大多数情况下，建议通过 `CUDA_VISIBLE_DEVICES` 环境变量来要设置使用的 GPU 设备。

* `torch.cuda.stream(stream)` - 将给定的 `stream` 用上下文管理器 `StreamContext` 包装
  * `stream` - 给定的流 `stream`，如果为 `None` 则该上下文管理器不会成为运算符

#### 1.5.1 随机数生成器

#### 1.5.2 交流集

#### 1.5.3 流和事件

* `torch.cuda.Stream` - CUDA 流的包装

  一个 CUDA 流是一个 GPU 操作队列，按照添加到流中的先后顺序依次执行，可以将一个流看做是 GPU 上的一个任务，不同任务可以并行执行。

  * `device (torch.device or int, optional)` - 分配流的设备，`None` 或负数时将会使用当前设备
  * `priority (int, optional)` - 流的优先级，可选 `-1`（高优先级）和 `0` 低优先级
  
  成员方法：
  
  * `wait_stream(stream)` - 与另一个流同步，提交到此流的所有未来工作将等待，直到所有核心在调用完成时被提交给给定的流
    * `stream (Stream)` - 要同步的流

### 1.6 `torch.optim`

`torch.optim` 是一个实现了多种优化算法的包。支持常见的优化算法，且接口充足有效。

**建立优化器：**

优化器的建立需要给出一个包含所需参数的可迭代对象。

```python
# 方法一：Construting it
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  	# 建立 SGD 优化器对象
optimizer = optim.Adam([var1, var2], lr=0.0001)  					# 建立 Adam 优化器对象

# 方法二：Per-parameter options
optim.SGD([{'params': model.base.parameters()},  					# model.base 的参数，使用默认学习率 lr = 1e-2
           {'params': model.classifier.parameters(), 'lr': 1e-3}  	# model.classifier 的参数，使用自定义学习率 lr = 1e-3
          ], lr=1e-2, momentum=0.9)  								# SGD 优化器的默认参数
```

【注1】如果要用 GPU，必须在建立优化器前，通过 `.cuda()` 将模型迁移到 GPU 上。

【注2】确保在建立优化器时，所有的优化参数都位于相同的设备上。

**优化器计算：**

所有优化器都实现 `step()` 方法，用于更新优化器的参数。

```python
# 方法一：optimizer.step()
for input, target in dataset:
    optimizer.zero_grad()  # 梯度清零
    output = model(input)  # 模型输出
    loss = loss_fn(output, target)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 优化器参数更新

# 方法二：循环 closure()（常用于多次重新估计函数的算法）
for input, target in dataset:
    def closure():
        optimizer.zero_grad()  # 梯度清零
        output = model(input)  # 模型输出
        loss = loss_fn(output, target)  # 计算损失
        loss.backward()  # 反向传播
        return loss
    optimizer.step(closure)  # 优化器参数更新
```

### 1.7 `torch.utils`

#### 1.7.5 `data`

这个文件包含了一些关于数据集处理的类。

![image-20211208111345432](img/image-20211208111345432.png)

* `torch.utils.data.Dataset` - 所有数据集的抽象基类，需要实现 `__getitem__` 和 `__len__` 函数
  * `__init__()` - 初始化该类的一些基本参数
  * `__getitem__()` - 迭代器传入索引，并期待得到数据集中相应的数据
  * `__len__()` - 给出整个数据集的尺寸大小，迭代器的索引范围由此而来
* `torch.utils.data.DataLoader(...)` - 数据加载器，组合数据集 Dataset 和采样器 Sampler，提供数据的 batch 迭代器
  * `dataset (Dataset)` – 需要加载的数据集（可以是自定义或者自带的数据集）
  * `batch_size` – batch 的大小（可选项，默认值为 `1`）
  * `shuffle` – 是否在每个 epoch 中 shuffle 整个数据集， 默认值为 `False`
  * `sampler` – 定义从数据中抽取样本的策略，如果指定了, shuffle 参数必须为 `False`
  * `batch_sampler` - 类似 `sampler`，但是一次性返回一个 batch 的下标
  * `num_workers` – 表示读取样本的线程数， `0` 表示只有主线程
  * `collate_fn` – 合并一个样本列表称为一个 batch
  * `pin_memory` – 是否在返回数据之前将张量拷贝到 CUDA
  * `drop_last (bool, optional)` – 设置是否丢弃最后一个不完整的 batch，默认为 `False`
  * `timeout` – 用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错，应该为非负整数

## 2 torchvision

torchvision library 是 PyTorch project 的一部分，它包含**计算机视觉领域**中热门的数据集、模型和常用图像变换工具。

### 2.1 数据集

torchvison 的数据集位于 **`torchvision.datasets`** 内。

所有的数据集都是 `torch.utils.data.Dataset` 的子类，因而都**重写**了 `___getitem__` 和 `__len__` 方法。于是，它们都可以**传入** `torch.utils.data.DataLoader` 中，通过 `torch.multiprocessing` workers **并行**读取多个样本。

**使用方法**：

```python
imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```

### 2.2 输入输出

torchvision 的**输入输出操作**位于 **`torchvision.io`** 内。

它们目前专门用于读写视频和图像。

### 2.3 计算机视觉模型

torchvision 的**计算机视觉模型**位于 **`torchvision.models`** 内。

其中包含用于处理不同任务的**模型定义**，包括：图像分类（image classification）、像素语义分割（pixelwise semantic segmentation）、对象检测（object detection）、实例分割（instance segmentation）、人物关键点检测（person keypoint detection）和视频分类（video classification）。

```python
# 模型创建
vgg16 = models.vgg16()
vgg16 = models.vgg16(pretrained=True)  # 使用预训练模型
```

#### 2.3.1 图像分类

#### 2.3.2 语义分割

#### 2.3.3 对象检测

##### VGG

`torchvision.models` 提供的 VGG 模型有 `VGG11`，`VGG13`，`VGG16`，`VGG19`，下面以 `VGG16` 为例进行介绍。

* `torchvision.models.vgg16(pretrained: bool = False, progress: bool = True, **kwargs) → torchvision.models.vgg.VGG`
  * VGG 16 层模型，出自 *[Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)*
  * 参数：
    * `pretrained` - `bool`，为真时返回在 ImageNet 上预训练的模型
    * `progress` - `bool`，为真时显示进度条
  * 返回值：
    * `torchvision.models.vgg.VGG` 模型
* `torchvision.models.vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs) → torchvision.models.vgg.VGG`
  * VGG 16 层模型，带批正则化，出自 *[Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)*
  * 参数：
    * `pretrained` - `bool`，为真时返回在 ImageNet 上预训练的模型
    * `progress` - `bool`，为真时显示进度条
  * 返回值：
    * `torchvision.models.vgg.VGG` 模型

#### 2.3.4 视频分类

### 2.4 计算机视觉操作

torchvision 的 **`torchvision.ops`** 实现特定的**计算机视觉操作**。

* `torchvision.ops.nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) → torch.Tensor`
  * 基于 intersection-over-union (IoU) 进行 non-maximum suppression (NMS) 操作
  * 参数：
    * `boxes` - `Tensor[N, 4]`，用于执行 NMS 的 boxes，形状为 `(x1, y1, x2, y2)`
    * `scores` - `Tensor[N]`，每个 boxes 的 scores
    * `iou_thredshold` - `float`，丢弃 IoU 大于此阈值的、重叠的 boxes
  * 返回值：
    * `keep` - `Tensor`，NMS 保留下来的那些 boxes 坐标，按照 scores 降序排列

### 2.5 图像变换操作

torchvision 的 **`torchvision.transforms`** 包含常用的**图像变换操作**。

它们可以通过 `torchvision.transforms.Compose` 进行**链式整合**，也可以通过 `torchvision.transforms.functional` 进行**更细粒度的变换**。

所有的 `transforms` 子类都接受以下两种类型的图像：

* `PIL Image`
* `Tensor Image`
  * `shape`：`(C, H, W)`，其中 `C` 是通道（channel）数，`H` 是图像高度（height），`W` 是图像宽度（width）
* `batch of Tensor Images`
  * `shape`：`(B, C, H, W)`，其中 `B` 是 `batch` 中包含的图像数量，其余同上
  * 应用于一批张量图像的确定性或随机变换相同地变换该批的所有图像

【注】所有随机性设定都使用 torch 默认的随机数生成器，手动设定方法如下：

```python
# Previous versions (<0.8)
# import random
# random.seed(12)

# Now
import torch
torch.manual_seed(17)
```

### 2.6 其他工具

torchvision 的**其他图像处理工具**位于 **`torchvision.utils`**。

## 3 辅助模块

### 3.1 PyTorch Metric Learning

深度度量学习有非常广泛的应用，但是这些算法的实现却乏味耗时，于是 PyTorch Metric Learning 应运而生，帮助研究者和应用者解决这个困难。

PyTorch Metric Learning 采用模块化、弹性化的设计，共包含如下 9 个 modules，彼此之间相互独立，也可以结合在一起使用。

<img src="img/high_level_module_overview.png" alt="high_level_module_overview.png" style="zoom: 50%;" />

Git 文档：https://github.com/KevinMusgrave/pytorch-metric-learning

