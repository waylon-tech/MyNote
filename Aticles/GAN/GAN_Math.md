## 目录

[toc]

## 1 GAN_Math

### 1.1 介绍

题目：A Mathematical Introduction to Generative Adversarial Nets（GAN）

论点：本文以数学角度叙述了目前（2020）经典 GAN 的基本原理

### 1.2 思想

**问题**

我们如何利用数据人工生成相似的对象？更明确的说，如何量化群体内对象的一致性 / 相似性？

**建模**

设对象为 $\mathbb{R}^n$ 空间中的一个点，于是给定数据集 $\mathcal{X} \in \mathbb{R}^n$ 及其上的分布 $\mu$，有

* 对象一致性：样本 $x$ 和 $y$ 均来自于 $\mathbb{R}^n$ 上的概率分布 $u$
* 对象相似性：样本 $x$ 和 $y$ 所属概率分布<u>相同或相近</u>

* 概率密度函数：假设概率分布 $u$ 拥有概率密度函数 $p(x)$，尽管其可能是退化分布[^1]，但实践中并不致命

因此，GAN 的目标就是**利用有限的数据集 $\mathcal{X}$ 生成分布 $v$，尽可能逼近真实分布 $\mu$**，因此其中的难度取决于：

* 分布 $\mu$ 的复杂性 - 分布越复杂难度越高
* 训练集 $\chi$ 的大小 - 数据集越大难度越小

[^1]: 退化分布 - 真实分布的维度小于密度函数表达的维度。

### 1.3 $v$-GAN

v-GAN 是 Vanilla GAN 的缩写，它是最基础、最经典的生成式对抗网络，因而以此作为 GAN 思想的介绍。

#### 1.3.1 数学模型

> **符号对照表**
>
> 分布
>
> * $\mu$：样本真实分布，维度 $\mathbb{R}^n$
> * $\gamma$：网络初始分布，维度 $\mathbb{R}^d$
> * $v$ : 初始分布的变换分布，定义为 $\gamma \circ G^{-1}$，维度 $\mathbb{R}^n$
>
> 样本
>
> * $x\in \mathbb{R}^n$ : 取自分布 $\mu$ 的样本
> * $z \in \mathbb{R}^d$ : 取自分布 $\gamma$ 的样本
> * $y \in \mathbb{R}^n$ : 变换变量，等于 $G(z)$，取自分布 $v$ 的样本
>
> 函数
>
> * $G_{\theta}$ : 生成函数，$\mathbb{R}^d \longrightarrow \mathbb{R}^n$，使用神经网络时的参数为 $\theta$
>
> * $D_{\omega}$ : 分类函数，$\mathbb{R}^n \longrightarrow [0,1]$，使用神经网络时的参数为 $\omega$
>
> 数据集
>
> * $\mathcal{X}$：训练数据集，分布为 $\mu$
>
> * $\mathcal{A}$：数据集，是 $\mathcal{X}$ 的子集
>
> * $\mathcal{B}$：样本集，取自分布 $\gamma$
>
> 【注1】$G^{-1}$ 表示将 $\mathbb{R}^n$ 中的子集映射到 $\mathbb{R}^d$。
>
> 【注2】$\gamma \circ G^{-1}$ 表示分布 $\gamma$ 的部分经 $G^{-1}$ 映射后得到的分布。

设 $\mu$ 为 $\mathbb{R}^n$ 上的真实分布，$\gamma$ 为 $\mathbb{R}^d$ 上的初始分布。GAN 目标是寻找一个映射 $G: \; \mathbb{R}^d \longrightarrow \mathbb{R}^n$，使得 $\forall z \sim \gamma$，$G(z)$ 的变换分布 $v := \gamma \circ G^{-1}$ 等于或者近似真实分布 $ \mu$。

为了找到这个函数 $G(z)$，GAN 引入判别函数 $D(x)$ 来辨认来自<u>真实分布 $\mu$</u> 的样本（正例）和<u>变换分布 $v$</u> 的样本（负例），从而形成生成函数 $G(z)$ 和判别函数 $D(x)$ 的对抗，这种博弈可以描述为对损失函数的最小最大化问题：
$$
\min_{G} \max_{D} V(D,G) := \min_{G} \max_{D} \left ( \mathbb{E}_{x \sim \mu}\left [ \log_{}{D(x)} \right ] + \mathbb{E}_{z \sim \gamma} \left [ \log_{}(1-D(G(z))) \right ] \right )
$$
其中的 ”**博弈**“ 表现在：

* 生成函数 $G$ 试图从分布 $\gamma$ 中产生以假乱真（逼近分布 $\mu$）的样本 $G(z)$
* 判别函数 $D$ 努力将假样本 $G(z)$ 从真样本 $x$ 中辨别出来

另外，令 $y=G(z) \in \mathbb{R}^n \sim v \sim \gamma \circ G^{-1}$ 统一积分限为 $\mathbb{R}^n$，设分布函数 $\mu(x)$ 和 $v(y)$ 有密度函数 $p(x)$ 和 $q(x)$ 统一积分元为 $dx$（要求 $d \ge n $），并忽略约束条件 $v = \gamma \circ G^{-1}, \exist G$，就可以得到化简的形式：
$$
\min_{G} \max_{D} V(D,G) := \min_{G} \max_{D} \int_{\mathbb{R}^n}^{} \left [ \log_{}{D(x)p(x)} + \log_{}{(1-D(x))q(x)} \right ]dx
$$

关于解，有定理（<u>定理 ==2.2==, ==2.3==</u>）保证，该最小最大问题的解为 $v=\mu$ 或 $q(x) = p(x)$，$D(x) = \frac{1}{2}$。此时，即使是分类器 $D(x)$ 达到最佳，其效果也不会比随机猜测更好。

#### 1.3.2 求解算法

在实践中，可以使用交替优化的算法思想进行求解，而公式中的总体均值使用样本均值代替。

**(1) 网络化**

使用神经网络实现生成函数和判别函数，得到 $D = D_\omega$，$G = G_\theta$ 和 $v_\theta = \gamma \circ G_\theta^{-1}$，于是最小最大问题变为
$$
\min_{\theta} \max_{\omega} V(D_\omega,G_\theta) := \min_{\theta} \max_{\omega} \int_{\mathbb{R}^n}^{} \left [ \log_{}{D_\omega(x)p(x)} + \log_{}{(1-D_\omega(x))q(x)} \right ]dx
$$
**(2) 离散化**

设 $\mathcal{A}$ 为训练集 $\mathcal{X}$ 的子集（分布为 $\mu$），$\mathcal{B}$ 为取自分布 $\gamma$ 的样本集，进行如下近似：
$$
\mathbb{E}_{x \sim \mu}\left [ \log D_\omega (x) \right ] \approx \frac{1}{\left | \mathcal{A} \right | } \sum_{x \in \mathcal{A}}^{} \log D_\omega (x)
$$

$$
\mathbb{E}_{z \sim \gamma}\left [ \log (1- D_\omega (G_\theta (z))) \right ] \approx \frac{1}{\left | \mathcal{B} \right | } \sum_{z \in \mathcal{B}}^{} \log (1- D_\omega (G_\theta (z)))
$$

**(3) 算法**

> **Vanilla GAN Algorithm** —— GAN 的小批量随机梯度下降求解算法
>
> 下面的 $k$ 是超参数。
>
> ```python
> for n in iterations: # 迭代次数
>       for k in steps: # 判别器优化次数（判别器不会直接训练到最优，而是每次改进一小点）
>            (1) 从分布 r 中抽取小批量的 m 个 Rd 维样本 {z1, ..., zm}
>            (2) 从数据集 X 中抽取小批量的 m 个 Rn 维样本 {x1, ..., xm}
>            (3) 更新判别器 D 的参数 omega (公式1)
>       [1] 从分布 r 中抽取小批量的 m 个 Rd 维样本 {z1, ..., zm}
>       [2] 更新生成器 G 的参数 theta (公式2)
> ```
>
> 注：随机梯度下降算法的梯度更新公式
> $$
> \bigtriangledown _w \frac{1}{m} \sum_{i=1}^{m} \left [ \log D_w(x_i) + \log(1-D_w(G_\theta (z_i))) \right ] \tag{公式 1}
> $$
>
> $$
> \bigtriangledown _w \frac{1}{m} \sum_{i=1}^{m} \log(1 - D_w(G_\theta(z_i))) \tag{公式 2}
> $$
>
> 该梯度更新公式可以在任何标准的梯度学习规则上使用。

【注】"$log \space D \space \mathcal{trick}$"：为避免在最小化 $\mathbb{E}_{z \sim \gamma}\left [ \log (1- D_\omega (G_\theta (z))) \right ]$ 时，函数 $G_\theta$ 更新过程中过早饱和，简化为 $-\mathbb{E}[\log D_\omega(G_\theta(z))]$。

### 1.4 $f$-GAN

GAN 使用概率分布建模数据的 “风格”，并通过有限的样本来学习其背后的概率分布。如何衡量概率分布的差异，能产生了不同的 GAN，表现在目标函数的不同。$f$-GAN 是 $f$-divergence GAN 的缩写，使用了 $f$ 散度作为 GAN minmax 的目标。

#### 1.4.1 散度

##### 散度定义

散度是度量两种概率分布相似性的指标。

* $KL$ 散度
  $$
  D_{KL}(p||q):=\int_{\mathbb{R}^n}^{}\log \left ( \frac{p(x)}{q(x)} \right ) p(x) dx
  $$

* $JS$​ 散度
  $$
  D_{JS}(p||q):=\frac{1}{2}D_{KL}(p||M)+\frac{1}{2}D_{KL}(q||M), \space M=\frac{p(x)+q(x)}{2}
  $$
  

* $f$ 散度

  令 $f(x)$ 为定义在 $I \in \mathbb{R}$ 上的严格凸函数，满足 $f(1)=0$，并设 $f(x \notin I)=+\infty$ .

  令 $p(x)$ 和 $q(x)$ 为 $\mathbb{R}^n$ 上的两个概率密度函数，则它们的 $f$ 散度为：
  $$
  D_{f}(p||q):=\mathbb{E}_{x\sim q}\left [ f\left ( \frac{p(x)}{q(x)} \right ) \right ] = \int_{\mathbb{R}^n}^{} f\left ( \frac{p(x)}{q(x)} \right ) q(x)dx
  $$
  其中规定 $q(x)=0$ 时 $f\left ( \frac{p(x)}{q(x)} \right ) q(x)=0$.

  【注1】$D_{KL}$ 和 $D_{JS}$ 都是 $f$ 散度的特例。

  【注2】期望的下标 $x\sim q$ 表示随机变量 $x$ 取遍 $\mathrm{supp}(q)$。

##### 散度性质

这里主要讨论更一般的 $f$ 散度的性质：

* 如果 $\mathrm {supp}(p) \subseteq \mathrm{supp}(q)$ 或 $f(t \in [0,1)) > 0$，则 $D_{f}(p||q)>0$ 且 $D_{f}(p||q)=0$ 当且仅当 $p(x)=q(x)$（==命题 3.1==）

* $f$ 散度可以推广到概率空间 $\Omega$ 上的两个任意概率分布 $\mu$ 和 $v$ 上，而不仅仅是概率密度函数

  令 $\tau$ 为另外的概率分布，且满足 $\mathrm{supp}(u,v) \subseteq \mathrm{supp}(\tau)$，令 $p=\frac{d\mu}{d\tau}$，$q=\frac{dv}{d\tau}$，则 $f$ 散度定义为
  $$
  D_{f}(\mu||v):= \int_{\Omega}^{} f\left ( \frac{\frac{d\mu}{dx} }{\frac{dv}{dx}} \right ) \frac{d\mu}{dx} dx = \int_{\Omega}^{} f\left ( \frac{p(x)\frac{d\tau}{dx} }{q(x)\frac{d\tau}{dx}} \right ) q(x) \frac{d\tau}{dx} dx = \int_{\Omega}^{} f\left ( \frac{p(x)}{q(x)} \right ) q(x) d\tau(x)  = \mathbb{E}_{x\sim v}\left [ f\left ( \frac{p(x)}{q(x)} \right ) \right ]
  $$
  同样规定 $q(x)=0$ 时 $f\left ( \frac{p(x)}{q(x)} \right ) q(x)=0$.

由于 $\mu$ 的分布未知，导致散度定义中的 $p(x)$ 难以计算，因此需要用有限的样本进行估计。下面用到凸共轭的方法，整体思路是将 $p(x)$ **藏在期望**中，然后用**样本均值代替总体均值**。

#### 1.4.2 凸共轭

##### 凸共轭定义

令 $f(x)$ 为定义在区间 $I \in \mathbb{R}$ 上的凸函数，它的凸共轭 $f^* : \mathbb{R} \longrightarrow \mathbb{R} \cup \left \{ \pm \infty \right \}$ 定义为
$$
f^*(y) = \mathrm{\underset{t\in I}{sup} \left \{ ty-f(t) \right \} }
$$
如果 $f(x)$ 严格凸且在 $I \in \mathbb{R}$ 上连续可导，令 $I$ 的开区间为 $I^o=(a,b), \space a,b \in [-\infty,+\infty]$，则有更明确的定义（==引理 3.2==）
$$
f^*(y)=\left\{\begin{matrix}
 yf^{'-1}(y)-f(f^{'-1}(y)), & y\in f^{'}(I^o) \\
 \lim_{t \to b^-} (ty-f(t)), & y \ge \lim_{t \to b^-}f^{'}(t) \\
 \lim_{t \to a^+} (ty-f(t)), & y \le \lim_{t \to a^+}f^{'}(t)
\end{matrix}\right.
$$
##### 凸共轭性质

* $f^*$ 是凸函数，并且下半连续；又由下半连续性，可推得对偶性 $f=(f^*)^*$

一些常用凸函数的**凸对偶**如下图所示。

<img src="img\GAN-GAN_Math-1.png" alt="GAN-GAN_Math-1" style="zoom: 67%;" />

#### 1.4.3 散度的凸共轭

令 $\mu, v, \tau$ 为概率分布，满足 $\mathrm{supp}(u,v) \subseteq \mathrm{supp}(\tau)$，并且 $p=\frac{d\mu}{d\tau}$，$q=\frac{dv}{d\tau}$.

令 $f(t)$ 为严格凸且在 $I \in \mathbb{R}$ 上连续可导函数，博雷尔概率函数 $\mu, v$ 满足 $\mathrm{supp}(\mu) \subseteq \mathrm{supp}(v)$，则（==定理 3.4==）
$$
\begin{align*}
D_{f}(p||q):= & \int_{\Omega}^{} f\left ( \frac{p(x)}{q(x)} \right ) q(x) d\tau(x) \\
 = & \int_{\Omega}^{} \underset{T}{\mathrm{sup}} \left \{ t\frac{p(x)}{q(x)} -f^*(t) \right \} q(x) d\tau(x) \\
 = & \int_{\Omega}^{} \underset{T}{\mathrm{sup}} \left \{ t p(x) -f^*(t)q(x) \right \} d\tau(x) \\
 = & \underset{T}{\mathrm{sup}}\left \{ \mathbb{E}_{x\sim\mu}[T(x)]-\mathbb{E}_{x\sim v}[f^*(T(x))] \right \} 
\end{align*}
$$
其中 $T$ 取遍所有博雷尔函数 $T\space:\space \mathbb{R}^n \longrightarrow \mathrm{Dom}(f^*)$.

【记】函数 $T(x)$ 是固定随机变量（样本） $x$ 时，取函数 $f(t)$ 的共轭过程（可看做另一种函数）中产生的，**内含“共轭”关系**。

【注1】这里的函数 $T(x)$ 正好起这判别器的作用，称为判别函数。

【注2】有定理指出，当 $f$ 在定义域无上界时，**条件 $\mathrm{supp}(\mu) \subseteq \mathrm{supp}(v)$**是必须的，否则恒等式不成立。（==定理 3.5==）

【注3】有定理指出，**低维流形数据**的情况下仍然能够获得好的结果（公式略），只是多了个常数项。（==定理 3.6==）

#### 1.4.4 VDM 算法

##### $f$-GAN 模型

有了离散的 $f$ 散度，就可以建立基于 $f$ 散度的 GAN 模型。对于一个给定的概率分布 $\mu$，$f$-GAN 的目标是基于概率分布 $v$ 最小化 $f$ 散度 $D_{f}(p||q)$：
$$
\min_{v}D_{f}(p||q) = \min_{v}\underset{T}{\mathrm{sup}} \left \{ \mathbb{E}_{x\sim\mu}[T(x)]-\mathbb{E}_{x\sim v}[f^*(T(x))] \right \}
$$
【注1】由定理 3.5，对 $\mathrm{supp}(\mu) \subseteq \mathrm{supp}(v)$ 条件不满足情形，若有**特定的 $f$** ：连续可导且在 $(0,1)$ 恒正，则仍能保证解存在且最优。（==定理 3.7==）

【注2】对于一些难以求共轭的函数，不要忘了还可以使用 **$f$ 散度的原始形式**。

##### 算法化处理

判别函数 $T(x)$ 来自神经网络 $T(x) = T_\omega(x) = g_f(S_\omega(x))$；

其中，

* $S_\omega$ 是神经网络，参数为 $\omega$，映射 $\mathbb{R^n} \longrightarrow \mathbb{R}$
* $g_f$ 是激活函数，映射 $\mathbb{R} \longrightarrow I^*$（$f^*$ 的定义域，即 $f'$ 的值域）

分布 $v$（$x, \space \mathbb{R}^n$ 维）用于生成假样本，仍然将其视为初始分布 $\gamma$（$z, \space \mathbb{R}^d$ 维）生成的 $v_\theta := \gamma \circ G_{\theta}^{-1}$；

生成函数 $G_\theta$ 是神经网络，参数为 $\theta$，映射 $\mathbb{R}^d \longrightarrow \mathbb{R}^n$；

综上，得到较为明确的 $f$-GAN 模型：
$$
\min_{v}\underset{\omega}{\mathrm{sup}} \left \{ \mathbb{E}_{z\sim \gamma}[g_f(S_\omega(G_\theta(z)))]-\mathbb{E}_{x\sim\mu}[f^*(g_f(S_\omega(x)))] \right \}
$$
##### 离散化处理

用样本均值代替总体均值。
$$
\mathbb{E}_{z\sim \gamma}[g_f(S_\omega(G_\theta(z)))] \approx \frac{1}{|\mathcal{B}|} \sum_{z\in\mathcal{B}}^{}\left [ g_f(S_\omega(G_\theta(z))) \right ]
$$

$$
\mathbb{E}_{x\sim \mu}[g_f(S_\omega(x))] \approx \frac{1}{|\mathcal{A}|} \sum_{z\in\mathcal{A}}^{} f^*(g_f(S_\omega(x)))
$$

##### 求解算法

>**VDM Algorithm** —— GAN 的小批量随机梯度下降求解算法
>
>下面的 $k\ge 1$ 和 $m$ 是超参数。
>
>```python
>for n in iterations: # 迭代次数
>      for k in steps: # 判别器优化次数（判别器不会直接训练到最优，而是每次改进一小点）
>             从分布 r 中抽取小批量的 m 个 Rd 维样本 {z1, ..., zm}
>             从样本集 X 中抽取小批量的 m 个 Rn 维样本 {x1, ..., xm}
>             更新神经网络 S (公式3)
>      从分布 r 中抽取小批量的 m 个 Rd 维样本 {z1, ..., zm}
>      更新生成器 G (公式4)
>```
>
>注：随机梯度下降算法的梯度更新公式
>$$
>\bigtriangledown_\omega \frac{1}{m} \sum_{i=1}^{m} \left [ g_f(S_\omega(G_\theta(z_i)))-f^*(g_f(S_\omega(x_i))) \right ] \tag{公式 3}
>$$
>
>$$
>\bigtriangledown_\theta \frac{1}{m} \sum_{i=1}^{m}g_f \left ( S_\omega(G_\theta(z_i)) \right ) \tag{公式 4}
>$$
>
>该梯度更新公式可以在任何标准的梯度学习规则上使用。

### 1.5 $w$-GAN

$w$-GAN 是 Wasserstein GAN 的缩写，它使用了 Wasserstein 距离（又称 EM 距离）作为 GAN minmax 的目标。

#### 1.5.1 失效模式

在训练 GAN 的过程中，常会遇到以下问题：

* 梯度消失
  * 当判别器性能过好时容易发生，导致生成器没有足够的信息（梯度）来提升自己
* 模式坍塌
  * 生成器的输出局限在一个局部区域，陷入局部最优
* 收敛失败
  * 原因是多方面的（已知和未知都有）

为了解决/缓和上述问题，提出了 $w$-GAN.

#### 1.5.2 EM 距离

EM 距离又称 Wasserstein-1 距离。

设 $\mu$, $v$ 是 $\mathbb{R}^n$ 上的任意两个分布，令 $\Pi(\mu,v)$ 表示 $\mathbb{R}^n\times\mathbb{R}^n$ 上随机变量 $\pi(x,y)$ 的分布，它的边际分布为 $\mu(x)$, $v(y)$，两者独立。则分布 $\mu$ 和 $v$ 的 EM 距离：
$$
W^1(\mu,v):=\min_{\pi\in\Pi(\mu,v)} \int_{\mathbb{R}^n\times\mathbb{R}^n}^{} \left \| x-y \right \| d\pi(x,y) = \min_{\pi\in\Pi(\mu,v)} \mathbb{E}_{(x,y)\sim\pi}\left [ \left \| x-y \right \|  \right ] 
$$
这种独立有着诸多优势，不详述。

#### 1.5.3 估计 $W^1$

和 $f$-GAN 的凸共轭处理一样，目标分布 $\mu$ 未知，因而 $\pu(x,y)$ 难以求得，需要进行估计。

这里直接给出结论，基于 Kantorovich-Rubenstein Duality：
$$
W^1(u,v) = \underset{T\in Lip_1(\mathbb{R}^n)}{\mathrm{sup}}\left ( \mathbb{E}_{x\sim\mu}[T(x)] - \mathbb{E}_{x\sim v}[T(x)] \right )
$$
然后的步骤同样类似于 $f$-GAN。

$w$-GAN 的目标函数是
$$
\min_{v}W^1(u,v) =\min_{v}  \underset{T\in Lip_1(\mathbb{R}^n)}{\mathrm{sup}}\left ( \mathbb{E}_{x\sim\mu}[T(x)] - \mathbb{E}_{x\sim v}[T(x)] \right )
$$
取 $v = \gamma \circ  G^{-1}$，使得
$$
\mathbb{E}_{x\sim v}[T(x)] = \mathbb{E}_{z\sim\gamma}[T(G(z))]
$$
最后，用神经网络近似函数 $T(x)$ 和 $G(x)$，参数分别为 $\omega$ 和 $\theta$，使用随机梯度下降算法求解，代码与上面的几乎一样。

#### 1.5.4 最后的问题

如何保证 $T\in Lip_1(\mathbb{R}^n)$？

* 原作者的做法是“权重裁剪”，限制 $\omega$ 在 $\Omega:={||\omega_\infty|| \le 0.01}$

* 根据 $\Omega$ 的紧凑性，$T_\omega : \omega \in \Omega$ 也会较为紧凑，因此可以合理近似
  $$
  \underset{\omega\in\Omega}{\mathrm{sup}}\left ( \mathbb{E}_{x\sim\mu}[T(x)] - \mathbb{E}_{x\sim v}[T(x)] \right )  \approx  \underset{T\in Lip_K(\mathbb{R}^n)}{\mathrm{sup}}\left ( \mathbb{E}_{x\sim\mu}[T(x)] - \mathbb{E}_{x\sim v}[T(x)] \right )
  $$

### 1.6 DC-GAN

DC-GAN 是 Deep Convolutional GAN 的缩写，它不是某种特定的 GAN，而是一些关于 GNA 的设计指导，能是训练更加稳定、结果更加显著。

* 用上采样和下采样代替所有的池化层
* 在生成器和判别器中使用分批正则化
* 不在深度结构中使用全连接隐层
* 在生成器中，除了输出层使用 Tanh 激活函数，其他所有层都使用 ReLU 激活函数
* 在判别器中，所有层都使用 LeakyReLU 激活函数

【注1】下采样：卷积核的移动距离超过一个单位。

【注2】上采样：卷积核的移动距离小于一个单位，这通过向输入向量添加 0，然后使用下采样来实现。

### 1.7 PG-GAN

PG-GAN 是 Progressive Growing of GANs 的缩写，是 GANs 的一种训练方法论。它的主要目的是解决高分辨率图像的生成问题，还顺带解决了目标分布奇异的情况（尤其是当原始分布与目标分布定义域不相交时）。

**思想概述**

* 训练生成低分辨率图像
* 逐步添加新的神经网络层，训练生成更高分辨率的图像

### 1.8 Cycle-GAN

Cycle-GAN 是 Cycle-Consistent Adversarial Networks 的缩写，它提供了一种更加广泛的图像风格迁移方法。

一个 Cycle-GAN 包含三种组件，对应三个定制的损失函数，给定两个训练数据集 $\mathcal{X}$ 和 $\mathcal{Y}$，Cycle-GAN 的目标是将 $\mathcal{X}$ 的风格迁移到 $\mathcal{Y}$（反之亦然）。

**(1) 组件一**

第一个组件是 GAN（常用 vanilla GAN，也可以是其他），目的是生成 $\mathcal{X}$ 的分布，其中特别的是**初始分布不再是的 $\gamma$，而是从 $\mathcal{Y}$ 中抽样得到**。

组件一的（对抗性）损失函数为：
$$
L_{\mathrm{gan1}}\left ( G_1,D_\mu \right ) := \mathbb{E}_{x\sim\mu}\left [ \log(D_\mu(x)) \right ] + \mathbb{E}_{y\sim v_0}\left [ \log(1-D_\mu(G_1(y))) \right ]
$$
**(2) 组件二**

第二个组件是组件一的镜像也就是说，它是 GAN，目的是生成 $\mathcal{Y}$ 的分布，其中初始分布从 $\mathcal{X}$ 中抽样得到。

组件二的（对抗性）损失函数为：
$$
L_{\mathrm{gan2}}\left ( G_2,D_{v_0} \right ) := \mathbb{E}_{y\sim v_0}\left [ \log(D_{v_0}(y)) \right ] + \mathbb{E}_{x\sim\mu}\left [ \log(1-D_{v_0}(G_2(x))) \right ]
$$
**(3) 组件三**

因为 `G(x)` 只需要符合域 `Y` 分布即可，并没有对其施加约束，所以 `x` 到 `G(x)` 包含很多种可能的映射。为此，作者提出使用循环一致性损失来作为约束，使得 `G` 生成的 `G(x)` 在内容上仍然能和 `x` 保持一致。

第三个组件是 “循环一致性” 损失函数：
$$
L_{\mathrm{cycle}}(G_1,G_2):=\mathbb{E}_{y\sim v_0}\left [ \left \| G_2(G_1(y))-y \right \|_1  \right ]+\mathbb{E}_{x\sim\mu}\left [ \left \| G_1(G_2(x))-x \right \|_1  \right ] 
$$
最后，总的损失函数为
$$
L^*\left ( G_1,G_2,D_\mu,D_{v_0} \right ) := L_{\mathrm{gan1}}(G_1,D_\mu)+L_\mathrm{gan2}(G_2,D_{v_0})+\lambda L_{cycle}(G_1,G_2)
$$
因此，Cycle-GAN 优化的目标是
$$
\min_{G_1,G_2} \max_{D_\mu,D_{v_0}} L^*(G_1,G_2,D_\mu D_{v_0})
$$

训练过程中的 Trick：

1. 学习率别太高！
2. 对抗损失权重不要太高，循环一致性损失权重为 `1` 的时候，对抗损失我一般设置为 `0.1`
3. 判别器优化频率高于生成器
4. 使用最小二乘损失

