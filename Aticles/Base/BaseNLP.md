## 目录

[toc]

## 1 一般架构

作为初始节，以文本分类为例介绍 NLP 的一般架构。

**(1) Embedding Layer**

目的：文本符号不能直接用于计算，因此需要映射为可以运算的数字。

方法：

* 给定单词组成的文本序列 $S = [w_1, w_2, \dots, w_L]$，将每个单词映射为一个 $d$ 维 embedding 向量，组成矩阵 $X = [x_1, x_2, \dots, x_L]$

* embeddings 取自 pretrained/learned embedding 模型，例如 Glove (Pennington et al., 2014)

特点：

* embedding 层只是简单地将符号映射为数字，向量系数，彼此独立

**(2) Encoding Layer**

目的：编码 embedding，得到蕴含时序、依赖信息的 encoding。

方法：

* Bi-LSTM
  $$
  \begin{align}
  h^f_t & = \text{LSTM}(h^f_{t-1}, x_t) \\
  h^b_t & = \text{LSTM}(h^b_{t+1}, x_t) \\
  h_t & = [h^f_t; h^b_t]
  \end{align}
  $$

  $$
  H = [h_1, h_2, \dots, h_L]
  $$

* BERT，见下文

特点：

* encoding 层编码 embedding 之间的依赖信息，得到有关联的 encoding

**(3) Aggregation Layer**

目的：将 encoding $H = [h_1, h_2, \dots, h_L]$ 聚集为固定个数和长度的 encoding $V = [v_1, v_2, \dots, v_M]$

方法：详见 `Base.md`

* Max or Average Pooling
* Self Attention Pooling
* Capsule Network Pooling

**(4) Prediction Layer**

将合适的 encoding 输出到 MLP 层得到 logits，再接一个 softmax 层得到概率，就能做出分类预测。
$$
p(·|\mathrm{e}) = softmax(\mathrm{MLP}(\mathrm{e}))
$$

## 2 语言模型

解释：用来计算一个句子的概率的模型，也就是判断一句话是否是人话的概率

定义：给定句子序列 $S = W_1,W_2,\dots,W_k$，它的语言模型 $P$ 的概率计算为
$$
P(S) = P(W_1,W_2,\dots,W_k) = p(W_1)p(W_2\mid W_1)\cdots P(W_k\mid W_1,W_2,\dots W_{k-1})
$$
其中概率 $p$ 由大规模语料库统计估算得到。

一般的语言模型有两个致命缺陷：

1. 參数空间过大：条件概率 $P(W_k\mid W_1,W_2,\dots W_{k-1})$ 的可能性太多，无法估算，不可能有用
2. 数据稀疏严重：对于非常多词对的组合，在语料库中都没有出现，依据最大似然估计得到的概率将会是 $0$

因此提出马尔可夫假设：

> 任意一个词出现的概率只与它前面出现的有限的一个或者 $n$ 个词有关。

如果 $n=0$，则一个词与周围的词是独立的，称为 unigram 即一元语言模型：
$$
P(S) = p(W_1)p(W_2)\cdots p(W_k)
$$
如果 $n=1$，则一个词只与前面一个词有关，称为 bigram 即二元语言模型：
$$
P(S) = p(W_1)p(W_2\mid W_1)p(W_3\mid W_2)\cdots p(W_k\mid W_{k-1})
$$
如果 $n=2$，则则一个词只与前面两个词有关，称为 trigram 即三元语言模型：
$$
P(S) = p(W_1)p(W_2\mid W_1)p(W_3\mid W_2, W_1)p(W_4\mid W_3, W_2)\cdots p(W_k\mid W_{k-1}, W_{k-2})
$$
一般 $n$ 超过 $4$ 的语言模型很少用到，因为它的上述两个缺陷将变得很明显。

【参考资料】

* [深入浅出讲解语言模型](https://zhuanlan.zhihu.com/p/28080127)