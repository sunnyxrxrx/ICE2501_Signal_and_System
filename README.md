# 《信号与系统》离散时间一维和二维卷积
### <font face=楷体 size=3>上海交通大学 电院2483 徐日晞 524030910245</font>
---
代码仓库： https://github.com/sunnyxrxrx/ICE2501_Signal_and_System.git
---
在《信号与系统》中的卷积和深度学习中的卷积有如下的区别
- 信号中的卷积的定义是
  $$y[n]=\sum_{i=-\infty}^{+\infty}x[i]h[n-i]$$
  在深度学习中卷积一般实现为
  $$y[n]=\sum_{i=-\infty}^{+\infty}x[i]h[n+i]$$
  严格上属于互相关运算。将卷积核翻转再做互相关运算，效果等价于直接卷积。只不过在深度学习中，卷积核不是给定的，而是需要学习的，因此我们直接采用更为简便的互相关运算。
- 在信号中，对于长度为$m$的样本和长度为$n$的卷积核，其输出的长度是$m+n-1$，也就是需要在输入两边pad上$(n-1)$的零元素，再让整个卷积核在输入滑动。
  
  在深度学习中，我们一般直接让卷积核在输入样本内滑动，输出长度为$m-n+1$，如果需要改变长度，需要通过增加padding参数实现。
## 卷积的程序实现
代码文件``my_conv.py``
### 一维卷积的程序实现
我们实现信号中的卷积。 定义输入、卷积核、输出均为一维数组，用numpy实现。

最原始的想法当然是套两层for循环，外层依次变量输出的下标$n$，内层计算每个下标的结果
```python
def my_1d(signal,kernel):
    len1=len(signal)
    len2=len(kernel)
    len3=len1+len2-1
    result=np.zeros(len3)
    signal=np.pad(signal,(len2-1,len2-1),"constant",constant_values=0)
    print(f"输出长度{len(result)}")
    for i in range(0,len3):
        for j in range (0,len2):
            result[i]+=kernel[len2-j-1]*signal[i+j]
    return result 
```
在python里面for循环是比较慢的，我们可以优化内层的计算，直接计算两个“窗口”的内积，得到如下改进的程序
```python
def my_1d_fast(signal,kernel):
    len1=len(signal)
    len2=len(kernel)
    len3=len1+len2-1
    kernel_flipped = kernel[::-1] 
    result=np.zeros(len3)
    signal_padded=np.pad(signal,(len2-1,len2-1),"constant",constant_values=0)
    print(f"输出长度{len(result)}")
    for i in range (0,len3):
        signal_window = signal_padded[i : i + len2]
        result[i]=np.dot(signal_window,kernel_flipped)
    return result   
```
我们用python自带的time库测试它们的性能，基于``1d_conv_demo.py``，稍作改动，得到``time_test_1d.py``，设计了两个卷积核，让它们去卷上课铃声，分别计算**卷1次**和**卷10次**的时间。多次运行取平均值。
```python
kernel1 = np.array([-1,2,-1],dtype=np.float64) 
kernel2=np.array([1/3, 1/3, 1/3],dtype=np.float64)
start_time=time.perf_counter()
for i in range(10):
    sharpened_signal = my_1d_fast(signal, kernel1)  
    down_signal=my_1d_fast(signal,kernel2)
end_time=time.perf_counter()
```
|实现方法|卷1次平均时间/s|卷10次平均时间/s|
|-------|---------|---|
|my_1d|1.81|18.22|
|my_1d_fast|**1.74**|**17.66**|

可见my_conv1_fast虽有优化，但效果不明显。

为进一步压榨性能，我们可以引入JIT即时编译。python是解释语言，每次都要动态检查传入的变量，在多次运行就会很慢，JIT能在程序首次执行时，将源代码**编译**成本地的机器码，并缓存结果。后续的调用将直接运行这个高度优化的机器码，从而极大地提升了执行速度。

我们可以用JIT装饰整个函数。它的要求是函数内的语言不能是过于复杂的语法（比如不能有try except，不能打印等），实测np.pad就编译不了，但是JIT对np.dot有比较好的适配，为此，手动实现padding
```python
from numba import jit
@jit(nopython=True)
def my_1d_jit(signal,kernel):
    len1=len(signal)
    len2=len(kernel)
    len3=len1+len2-1
    kernel_flipped=kernel[::-1] 
    result=np.zeros(len3)
    pad_width=len2-1
    signal_padded=np.zeros(len1+2*pad_width)
    signal_padded[pad_width:pad_width+len1]=signal
    for i in range(len3):
        signal_window=signal_padded[i:i + len2]
        result[i]=np.dot(signal_window, kernel_flipped)
    return result
```
|实现方法|平均时间/s|
|-------|---------|
|my_1d|18.22|
|my_1d_fast|17.66|
|my_1d_jit|**4.99**|
可以看到JIT在多轮运行时有显著优势

### 二维卷积的程序实现
输入和输出是用numpy储存的二维数组

最原始的实现当然还是for循环，4层循环，外层遍历每个位置，内层计算每个位置
```python
def my_2d(signal,kernel):
    lenm=np.size(signal,axis=0)# m是行数，n是列数
    lenn=np.size(signal,axis=1)
    m=np.size(kernel,axis=0)
    n=np.size(kernel,axis=1)
    signal=np.pad(signal,((m-1,m-1),(n-1,n-1)),'constant',constant_values=0)
    result=np.zeros((lenm+m-1,lenn+n-1))
    for i in range(0,lenn+n-1):
        for j in range(0,lenm+m-1):
            for x in range(0,m):
                for y in range(0,n):
                    result[i][j]+=signal[i+x][j+y]*kernel[m-x-1][n-y-1]
    return result
```
同样，用内积+JIT改进如下
```python
@jit(nopython=True)
def my_2d_jit(signal,kernel,keep=False):
    lenm,lenn=signal.shape # 信号的大小
    m,n=kernel.shape # 卷积核的大小
    assert (m%2==1 and n%2==1)
    result=np.zeros((lenm+m-1,lenn+n-1))
    signal_padded=np.zeros((lenm+2*m-2,lenn+2*n-2))
    signal_padded[m-1:m+lenm-1,n-1:n+lenn-1]=signal
    kernel_flipped = np.flip(kernel)
    for i in range(lenm+m-1):
        for j in range(lenn+n-1):
            signal_window = signal_padded[i : i + m, j : j + n]
            result[i, j] = np.sum(signal_window * kernel_flipped)
    if keep:
        return result[(m-1)//2:(m-1)//2+lenm,(n-1)//2:(n-1)//2+lenn]
    return result
```
## 卷积的pytorch实现和比较
测试代码文件 ``test_my_conv.py``
### 使用pytorch实现一维卷积
```python
def my_1d_torch(signal, kernel):
    kernel_flip = kernel[::-1].copy()
    kernel_tensor = torch.tensor(kernel_flip, dtype=torch.float64).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    signal_tensor = torch.tensor(signal, dtype=torch.float64).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    result = F.conv1d(signal_tensor, kernel_tensor, padding=kernel.shape[0]-1)
    return result.squeeze().detach().numpy()
```
注意这里pad是shape[0]-1不是shape[0]//2，因为我并不是想要输出和输入规模相同，而是遵照信号与系统中的定义。

### 测试结果一致性
```python
signal_test = np.array([1, 2, 3, 4], dtype=np.float64)
kernel_asymmetric = np.array([1, 2], dtype=np.float64) 
result_jit = my_1d_jit(signal_test, kernel_asymmetric)
result_torch = my_1d_torch(signal_test, kernel_asymmetric)
print("JIT结果:  ", result_jit)
print("PyTorch结果:", result_torch)
print("结果是否一致:", np.allclose(result_jit, result_torch))
```
输出相同。值得一提的是，使用pytorch的卷积10轮算下来只用0.03秒，说明手动实现的还是有很大优化空间的。
### 使用pytorch实现二维卷积
```python
def my_2d_torch(signal, kernel,keep=False):
    m,n=kernel.shape
    assert (m%2==1 and n%2==1)
    kernel_flip = np.flip(kernel).copy()
    kernel_tensor = torch.tensor(kernel_flip, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    signal_tensor = torch.tensor(signal, dtype=torch.float64).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    if keep:
        return F.conv2d(signal_tensor, kernel_tensor, padding=((m-1)//2,(n-1)//2)).squeeze().detach().numpy()

    return F.conv2d(signal_tensor, kernel_tensor, padding=(m-1,n-1)).squeeze().detach().numpy()
```
注：
在2D的卷积中，为了后面操作的需要，设置keep的选项，其为真时，将通过合适的padding将输出和输入设置为一样的size，同时为了padding的方便，保持对称性，我们断言卷积核的size都是奇数。

### 测试结果一致性
```python
signal2=np.random.rand(10,10)
kernel2=np.random.rand(3,3)
result_jit1 = my_2d_jit(signal2, kernel2,keep=False)
result_torch1 = my_2d_torch(signal2, kernel2,keep=False)
print(result_jit1)
print(result_torch1)
print("结果是否一致:", np.allclose(result_jit1,result_torch1))
```
分别置keep=True和False均通过检验。

---

<mark>
注：下面的操作我们分别调用前面自己实现的卷积函数和使用pytorch库的卷积函数
</mark>

## 一维语音信号的卷积
代码文件 ``1d_conv.py``

可以使用vscode的插件audio prewiew查看声音信号的波形图，梅尔谱等

原始波形
![Alt text](image-2.png)
### 锐化
$$ \begin{bmatrix}
-1& -1& 5&-1&-1
\end{bmatrix}$$
工作原理：
该卷积核将中间位置的信号放大，并抑制两侧的信号，也可以理解为是中间信号和两侧信号的差值，如果中间位置发生信号的突变，那么卷积后的响应值会较大；如果该段时间音频信号保持不变，那么卷积后将不会发生变化。

考察 
$$ \begin{bmatrix}
-1& -1& 5&-1&-1
\end{bmatrix}\cdot  \begin{bmatrix}
x_0 & x_0 &x_0 &x_0 &x_0
\end{bmatrix}=x_0$$
$$ \begin{bmatrix}
-1& -1& 5&-1&-1
\end{bmatrix}\cdot  \begin{bmatrix}
x_0 & x_0 &4x_0 &x_0 &x_0
\end{bmatrix}=16x_0$$
卷积核在计算时会突显音频信号的突变或高频成分，减弱平滑部分，增强音频的细节，使得音频听起来更加“锐化”。
![Alt text](image-3.png)
可以看到，经过锐化卷积后，音频信号的突变部分（铃声的每个deng的声音）被加强，表现出来铃声会更尖锐。
### 加噪和去噪
加噪不需要卷积，而是逐元素加上一个服从高斯分布的随机噪声，加噪后的波形可视化如图所示
![加噪后的波形](image-4.png)
可见波形在[-0.2,0.2]增加了更多噪声，表现在音频里就是更多背景杂音。
#### 均值卷积核
$$ \begin{bmatrix}
\frac{1}{n}& \frac{1}{n}& ...&\frac{1}{n}&\frac{1}{n}
\end{bmatrix}\in \mathbb{R}^{1\times n}$$
这相当于和前后的信号做平均。

- 对噪声: 由于我们加的是高斯噪声，它的特点是随机和独立，以均值为0采样。故在一定时间窗口内平均值也接近于零，从而取平均值能消去部分噪声。
- 对于原始信号: 在一个很小的局部窗口内，信号的变化是有趋势、有规律的。如果当前点的值是0.8，那么它旁边的点很可能也是0.8附近的值，比如0.79或0.81，它们的平均值仍然会非常接近0.8。所以，原始信号本身损失不大。
  
  当然，正如在前面锐化中提到的，如果原始音频中存在突变部分，那么平均卷积会把它削弱了。

  合理选择$n$也很重要，太大的$n$会把原始音频也平滑掉。下图中，上面为$n=10$的结果，可见噪声明显减少，和原始声音相似。下面为$n=1000$的结果。可见$n$太大会产生荒谬的结果，效果适得其反。

![10](image-8.png)
![Alt text](image-10.png)

#### 高斯卷积核
$$
G(x)=\frac{1}{\sqrt{2\pi}\sigma} {e^{-\frac{x^2}{2\sigma^2}}}$$
$x$为到中心点的距离，$\sigma$是参数

在实际构造时，我们一般先计算${e^{-\frac{x^2}{2\sigma^2}}}$，然后对其归一化即可。代码实现如下
```python
def gaussian_kernel(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)
```


原理：就像上面提到的，平均卷积会导致一个副作用：边缘模糊。当信号有一个急剧的变化时会把这个尖锐的边缘“平均掉”，使其变得平缓。

作为改进，高斯核引入加权平均，权重来自高斯分布。离中心点越近的采样点，与中心点的关系越密切，应该给予更高的权重，而远处点的权重很低。这能更好地保留边缘，使其不容易被模糊。

$\sigma$参数需要调整，下图分别展示了$\sigma=5,10,20$ 的结果，可见$\sigma=5$ 时效果较好。
![Alt text](image-7.png)
![Alt text](image-6.png)
![Alt text](image-5.png)

## 二维图像信号的卷积
代码文件 ``2d_conv.py``
### 加噪和去噪

#### 均值卷积核
最常见的 **3x3 均值核**:

$$
\frac{1}{9} \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$
就像一维卷积所提到那样，均值卷积在去噪同时，**会使图像的边缘和细节变得模糊**，如下图所示。
![Alt text](image-11.png)
#### 高斯卷积核 
进行加权平均，离中心像素越近的邻居权重越高，越远的权重越低。权重分布符合二维高斯函数。
$$ G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
例如，σ≈1、3x3的高斯核:
$$
K \approx \frac{1}{16} \begin{bmatrix} 
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix}
$$
去噪效果比均值滤波更平滑、自然，**对边缘的保留能力更好**。
### 边缘提取
边缘一般是像素突变的位点，我们可以用梯度大小衡量突变情况。

#### Sobel算子
$$ G_x=\begin{bmatrix}
-1 &0&1\\
-2 &0&2\\
-1& 0&1\\
\end{bmatrix}\ \  G_y=\begin{bmatrix}
-1 &0&-1\\
0 &0&0\\
1& 2&1\\
\end{bmatrix}$$
原理： $G_x$相当于在$x$方向求**一阶导**（中心差分），因而在水平方向有突变的位置会被强化，故用于检测垂直边缘；$G_y$是$G_x$的转置，用于检测竖直边缘。

下面我们用我在香港地铁拍的一张照片作为试验，原图如下
![Alt text](image-15.png)
垂直边缘检测
![Alt text](image-16.png)
水平边缘检测
![Alt text](image-17.png)
可见$G_x$和$G_y$分别对垂直和水平边缘做了强化。例如，垂直边缘核卷积后，图片的垂直线条，例如瓷砖上的竖线、“尖东”的竖画，得到了强化（白色部分）。
#### Laplace算子
$$L=\begin{bmatrix}
0 &1&0\\
1 &-4&1\\
0& 1&10\\
\end{bmatrix}$$
原理：它计算的是每个位置的二阶导数
$$
\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}
$$
一维情况下，二阶导的中心差分可以写成：
$$
\frac{\partial^2 f}{\partial x^2} = f(x + 1) - 2f(x)+ f(x - 1)
$$

$$
\frac{\partial^2 f}{\partial y^2} = f(y + 1) - 2f(y) + f(y - 1)
$$
那么得到上述二阶微分：
$$
\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = [f(x + 1, y) + f(x - 1, y) + f(x, y + 1) + f(x, y - 1)] - 4f(x, y)
$$
![Alt text](image.png)
可以看到Lapalce卷积后并没有提取的明显的边缘，效果并不好。

分析：Laplace是二阶导数，它寻找的是图像强度变化率的顶点。它在平坦区域和**线性渐变区域**的响应都接近于零，注意线性渐变在一阶导的Sobel算子是有响应的，但在Lapalce中没有。
而我拍的这个香港地铁的照片，大面积的灰色面板是平坦区域，Laplace算子在这些区域的输出就接近0。

文字“尖東”和英文字母的边缘虽然清晰，但在真实拍摄的图像中，它们总有1-2个像素的**平滑**过渡，导致二阶导数的响应值会非常小。
#### Canny算子
代码文件 ``canny.py``

Sobel算子只能提取水平或竖直方向的边缘，提取各方向边缘的卷积核有Laplace算子等，综合前面讲的各种方法，介绍Canny算子，能有效提取各方向的边缘

原始图像 → 降噪 → 求梯度 → 细化边缘 → 筛选边缘 → 最终结果

- 先用前面提到的高斯核去除图像噪声，避免噪声被误判为边缘，这会模糊图像，但保留主要结构
- 使用Sobel算子分别计算水平和垂直方向的梯度幅值和方向
  $$
  G = \sqrt{G_x^2 + G_y^2}
  $$
  $$
  \theta = \arctan\left(\frac{G_y}{G_x}\right)
  $$
  - 梯度幅值大 → 该点可能是边缘
  - 梯度方向 → 告诉我们边缘的朝向

- 非极大值抑制：沿着梯度方向，只保留局部最大值

  1. **将梯度方向量化为4个方向**
  2. **比较当前像素与梯度方向上的两个邻居**：
  例如：如果梯度方向是水平
  比较：左边像素 ← [当前] → 右边像素

  只有当当前梯度值 $\ge$ 左右邻居时才保留
  否则设为0


- 双阈值检测
使用两个阈值（$T_{high}$ 和 $T_{low}$）
    - 大于$T_{high}$ 直接保留
    - 小于$T_{low}$ 直接丢弃
    - 介于两者之间：检查其8邻域，如果邻域中有强边缘，保留，否则丢弃
        
![Alt text](image-19.png)
可见Canny算子能够提取各方向的边缘，形成完整的轮廓。

## 附录
与大模型的交互过程
- 1
    - User:
    ```
    def my_1d(signal,kernel):
    len1=len(signal)
    len2=len(kernel)
    len3=len1+len2-1
    result=np.zeros(len3)
    signal=np.pad(signal,(len2-1,len2-1),"constant",constant_values=0)
    print(len(result))
    for i in range(0,len3):
    for j in range (0,len2):
    result[i]+=kernel[len2-j-1]*signal[i+j]
    return result   这是我写的一维卷积的代码（按照信号与系统要求，翻转卷积核），怎么加快计算
    ```
    - Output:
    ```
          好的，这个要求非常明确，而且在学术和面试中很常见。目的是考察你是否理解卷积的底层实现，并能否在不使用“黑盒”函数的情况下进行优化。

      你的原始代码是正确的，但性能瓶颈在于双层`for`循环。我们的目标就是消除这些Python循环，或者让它们运行得更快。

      这里有三种主要的加速策略，从易到难，都完全符合你的要求。

      ---

      ### 方法一：向量化内层循环（最推荐）

      这是最核心、最“NumPy风格”的优化。你的内层循环实际上是在计算两个向量的**点积 (Dot Product)**。NumPy的点积运算是用高度优化的C语言或Fortran代码实现的，比Python循环快几个数量级。

      **思路：**
      1.  **预先翻转卷积核**：在循环开始前，就把`kernel`翻转一次，存入新变量。
      2.  **保留外层循环**：外层循环用于在信号上“滑动”窗口。
      3.  **用`np.dot`替换内层循环**：在每次外层循环中，从补零后的信号中取出一个与卷积核等长的“窗口”，然后用`np.dot()`计算这个窗口和翻转后卷积核的点积。

      **代码实现：**

      ```python
      import numpy as np

      def conv_optimized_v1(signal, kernel):
          # 1. 获取尺寸
          len_s = len(signal)
          len_k = len(kernel)
          len_r = len_s + len_k - 1
          
          # 2. 预先翻转卷积核 (这是关键的第一步)
          kernel_flipped = kernel[::-1] # 或者 np.flip(kernel)
          
          # 3. 对信号进行补零，为滑动窗口做准备
          #    只需要在信号的一侧补零即可
          signal_padded = np.pad(signal, (len_k - 1, len_k - 1), 'constant')
          
          # 4. 准备结果数组
          result = np.zeros(len_r)
          
          # 5. 只用一层循环来滑动窗口
          for i in range(len_r):
              # 从补零后的信号中提取当前窗口
              signal_window = signal_padded[i : i + len_k]
              
              # 6. 用np.dot()高效计算乘积之和，替换内层循环
              result[i] = np.dot(signal_window, kernel_flipped)
              
          return result
      ```

      **为什么这个更快？**
      你已经把 `len_k` 次Python循环操作（内层循环）替换成了一次对`np.dot`的调用。`np.dot`内部的循环是在编译好的C代码中执行的，速度飞快。

      ---

      ### 方法二：使用Numba进行即时编译（JIT）

      这是一种非常强大的技术。Numba是一个库，它可以获取你的Python代码（特别是包含循环和NumPy操作的代码），并将其即时编译（Just-In-Time, JIT）成速度可与C语言媲美的机器码。

      最棒的是，你几乎不需要修改你的原始代码，只需要在函数上加一个“装饰器”即可。

      **思路：**
      4.  安装Numba: `pip install numba`
      5.  导入`jit`装饰器。
      6.  在你原来的函数上加上`@jit(nopython=True)`。`nopython=True`告诉Numba，如果不能把所有代码都转成高效的机器码，就报错。这能确保你获得最佳性能。

      **代码实现：**

      ```python
      from numba import jit
      import numpy as np

      # 这就是你的原始代码，只加了一行装饰器！
      @jit(nopython=True)
      def conv_numba(signal, kernel):
          len1 = len(signal)
          len2 = len(kernel)
          len3 = len1 + len2 - 1
          result = np.zeros(len3)
          
          # 注意：Numba可能不支持np.pad的所有参数，我们手动实现补零
          # 创建一个更大的数组，然后把signal放进去
          signal_padded = np.zeros(len1 + 2 * (len2 - 1))
          signal_padded[len2 - 1 : len2 - 1 + len1] = signal

          for i in range(len3):
              for j in range(len2):
                  result[i] += kernel[len2 - j - 1] * signal_padded[i + j]
          return result
      ```

      **为什么这个更快？**
      Numba分析了你的双层循环，并为它们生成了专门优化的、类型固定的机器码。它消除了Python解释器的所有开销，使得循环可以全速在CPU上运行。

      ---
    ```
- 2
  - User:在音频去噪任务中，有什么1维卷积核可以选择？我只需要固定的卷积核，不是可学习的
  - Output：
    ```
      好的，完全明白！您要找的是经典的**数字信号处理（DSP）**中的**固定滤波器核（Fixed Filter Kernels）**，而不是深度学习中通过训练得到的权重。这些滤波器通过与一维时域波形进行卷积，来实现特定的去噪效果。

      这是一种非常经典且高效的去噪方法，尤其适用于特定类型的噪声。

      以下是您在时域波形上可以使用的主要几类固定卷积核/滤波器：

      ---

      ### 1. 基础平滑/低通滤波器 (Basic Smoothing / Low-pass Filters)

      这类滤波器的核心思想是“平均化”，假设噪声是高频的、随机的，而有用信号是相对平滑的。通过局部平均，可以抑制噪声，平滑信号。

      #### 1.1. 均值卷积核 (Moving Average Kernel)
      这是最简单、最直观的低通滤波器。

      *   **核心思想**: 将信号中每个点替换为其邻近N个点的平均值。
      *   **卷积核形式**: 一个长度为 N 的窗口，其中每个元素的值都是 `1/N`。
          *   例如，一个长度为 3 的均值卷积核是：`[1/3, 1/3, 1/3]`
          *   一个长度为 5 的均值卷积核是：`[0.2, 0.2, 0.2, 0.2, 0.2]`
      *   **擅长处理的噪声**: 高斯白噪声、背景嘶嘶声（hiss）。
      *   **优点**:
          *   计算极其简单、快速。
      *   **缺点**:
          *   会导致信号的**边缘和瞬态部分变得模糊**（比如鼓点、辅音的起音会变“软”）。
          *   在频域上，它的阻带衰减不够理想，会产生所谓的“旁瓣效应”。

      #### 1.2. 高斯卷积核 (Gaussian Kernel)
      这是均值滤波器的改进版，给予中心点更高的权重，距离中心越远的点权重越低。

      *   **核心思想**: 使用高斯函数（正态分布曲线）作为权重，进行加权平均。
      *   **卷积核形式**: 根据高斯函数 `G(x) = exp(-x² / (2σ²))` 生成。`σ` (标准差) 控制了平滑的程度，`σ` 越大，平滑效果越强，卷积核越宽。
          *   例如，一个近似的高斯核可能看起来像：`[0.05, 0.24, 0.4, 0.24, 0.05]` (数值需要归一化，使其总和为1)。
      *   **擅长处理的噪声**: 高斯白噪声。
      *   **优点**:
          *   平滑效果比均值滤波器更自然，对信号边缘的破坏更小。
          *   在时域和频域之间有很好的平衡（时频不确定性原理）。
      *   **缺点**:
          *   计算比均值滤波器稍复杂。
          *   仍然会模糊信号的细节。

      ---

      ### 2. 中值滤波器 (Median Filter) - (非线性，但概念相关)

      **严格来说，中值滤波不是一个线性卷积操作**，但它在概念上和应用上与卷积非常相似（都是在滑动窗口内操作），并且在去噪中极为常用和有效，所以必须提及。

      *   **核心思想**: 将信号中每个点替换为其邻近N个点的**中位数**。
      *   **操作方式**: 在一个滑动窗口内，对所有采样点进行排序，取中间的那个值作为输出。
      *   **擅长处理的噪声**: **脉冲噪声 (Impulse Noise)**，例如爆音（clicks）、噼啪声（pops）、椒盐噪声（salt-and-pepper noise）。
      *   **优点**:
          *   **在去除脉冲噪声方面效果极佳**。
          *   **能很好地保留信号的边缘**，不会像均值或高斯那样产生模糊。
      *   **缺点**:
          *   计算开销比线性卷积大（因为需要排序）。
          *   对于高斯噪声的抑制效果不如均值或高斯滤波器。

      ---

      ### 3. 频率选择性滤波器 (Frequency-Selective Filters)

      这类滤波器是在频域中设计的，但最终会表现为时域上的一个固定卷积核（FIR滤波器）。它们的目标是精确地保留或去除特定频段的信号。

      #### 3.1. 带阻/陷波滤波器 (Band-stop / Notch Filter)
      *   **核心思想**: 精确地去除一个或多个非常窄的频率带。
      *   **卷积核形式**: 通常通过算法（如窗函数法）设计，形式比较复杂，不是简单的`[a, b, c]`。
      *   **擅长处理的噪声**: **特定频率的嗡嗡声 (Hum/Buzz)**，最常见的是 50Hz 或 60Hz 的工频干扰及其谐波。
      *   **优点**:
          *   对目标频率的噪声去除非常彻底。
          *   对其他频段的信号影响极小。
      *   **缺点**:
          *   设计相对复杂，需要知道噪声的准确频率。

      #### 3.2. 带通滤波器 (Band-pass Filter)
      *   **核心思想**: 只允许特定频率范围内的信号通过，滤除该范围之外的所有信号。
      *   **擅长处理的噪声**: 频带超出目标信号范围的噪声。例如，在语音处理中，可以设计一个带通滤波器只保留 300Hz-3400Hz 的语音核心频段，滤除低频的隆隆声和高频的嘶嘶声。
      *   **优点**:
          *   可以有效地将目标信号从噪声中“分离”出来。
      *   **缺点**:
          *   如果目标信号的频谱很宽，可能会损失一部分有用信息。

      ---

      ### 总结与选择建议

      | 卷积核/滤波器类型 | 核心思想 | 擅长处理的噪声 | 优缺点 |
      | :--- | :--- | :--- | :--- |
      | **均值核** | 局部平均 | 高斯噪声、白噪声、嘶嘶声 | **优点**: 简单快速<br>**缺点**: 严重模糊信号边缘 |
      | **高斯核** | 加权局部平均 | 高斯噪声、白噪声 | **优点**: 平滑效果自然<br>**缺点**: 仍然会模糊信号 |
      | **中值滤波** (非线性) | 局部中位数 | **脉冲噪声** (爆音、噼啪声) | **优点**: **极好地保留边缘**<br>**缺点**: 计算稍慢，不适合高斯噪声 |
      | **带阻/陷波核** | 滤除特定频率 | **工频干扰** (50/60Hz嗡嗡声) | **优点**: 精准打击<br>**缺点**: 需知噪声频率 |
      | **带通核** | 保留特定频率 | 低频隆隆声、高频嘶嘶声 | **优点**: 分离信号与带外噪声<br>**缺点**: 可能损失有用信号 |

      ### 如何在实践中使用？

      你不需要手动去写这些卷积核的数值（除了最简单的均值核）。在 Python 中，可以使用 `SciPy` 和 `NumPy` 库来轻松实现：

      *   **线性卷积 (均值, 高斯, 带通/带阻)**:
          *   使用 `numpy.convolve` 或 `scipy.signal.lfilter` / `scipy.signal.filtfilt`。
          *   对于高斯核，可以用 `scipy.ndimage.gaussian_filter1d`。
          *   对于带通/带阻，可以用 `scipy.signal.firwin` 来设计卷积核（FIR系数），然后用 `convolve` 应用它。

      *   **中值滤波**:
          *   直接使用 `scipy.signal.medfilt`。

      **代码示例 (使用NumPy进行均值滤波):**

      ```python
      import numpy as np
      import matplotlib.pyplot as plt

      # 1. 生成一个带噪声的信号
      fs = 1000  # 采样率
      t = np.linspace(0, 1, fs, endpoint=False)
      # 纯净信号 (一个50Hz的正弦波)
      clean_signal = np.sin(2 * np.pi * 50 * t)
      # 添加高斯噪声
      noise = 0.5 * np.random.randn(len(t))
      noisy_signal = clean_signal + noise

      # 2. 定义一个均值卷积核
      window_size = 11  # 必须是奇数，以便有明确的中心
      mean_kernel = np.ones(window_size) / window_size

      # 3. 应用卷积进行去噪
      # mode='same' 使得输出数组和输入数组一样长
      denoised_signal = np.convolve(noisy_signal, mean_kernel, mode='same')

      # 4. 可视化结果
      plt.figure(figsize=(12, 6))
      plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.5)
      plt.plot(t, denoised_signal, label=f'Denoised (Moving Average, N={window_size})', linewidth=2)
      plt.plot(t, clean_signal, label='Clean Signal', linestyle='--', color='black')
      plt.legend()
      plt.title("Denoising with a Fixed Moving Average Kernel")
      plt.xlabel("Time [s]")
      plt.ylabel("Amplitude")
      plt.grid(True)
      plt.show()
      ```
- 3
  - User：除了Sobel算子外，还有什么边缘检测核，要求能识别各个方向的边缘
  - Output：当然！除了Sobel算子外，一种被认为是最优的边缘检测方法时Canny算子，它是一种多阶段的边缘检测算法，由John Canny在1986年提出。
  
    算法步骤
    ```
    ### 1. 高斯滤波降噪
    使用高斯滤波器平滑图像，减少噪声干扰：
      - 高斯核示例 (5×5, σ=1.4):
    [2  4  5  4  2]
    [4  9 12  9  4]
    [5 12 15 12  5] × 1/159
    [4  9 12  9  4]
    [2  4  5  4  2]


    ### 2. **计算梯度幅值和方向**
    使用Sobel算子计算图像梯度：

    Gx = Sobel_x * 图像  (水平梯度)
    Gy = Sobel_y * 图像  (垂直梯度)

    梯度幅值: G = √(Gx² + Gy²)
    梯度方向: θ = arctan(Gy/Gx)


    ### 3. **非极大值抑制 (Non-Maximum Suppression)**
    这是Canny的**关键创新**：
    - 沿着梯度方向，只保留局部最大值
    - 抑制非边缘点，使边缘变细到**单像素宽度**
    - 比较当前像素与梯度方向上的两个邻居

    **原理示意：**

    如果梯度方向是 →
    只有当前点的梯度值 > 左右邻居时才保留

      [小] [大] [小]  → 保留中间点
       ←   ↑   →
           梯度方向


    ### 4. **双阈值检测 (Double Threshold)**
    使用**两个阈值**分类边缘：
    - **强边缘**：梯度 > 高阈值 (high)
    - **弱边缘**：低阈值 (low) < 梯度 < 高阈值
    - **非边缘**：梯度 < 低阈值

    ### 5. **边缘连接 (Edge Tracking by Hysteresis)**
    - 保留所有**强边缘**
    - 只保留与强边缘**连接**的弱边缘
    - 丢弃孤立的弱边缘

    强边缘 ━━ 弱边缘 ━━ 强边缘  ✓ 保留
    弱边缘 (孤立)              ✗ 丢弃
    ```
- 4
  - User：帮我实现canny算子方法，我已经有了卷积的函数，名叫my_2d_jit(img_tensor.squeeze().detach().numpy() , kernel,keep=True)，直接调用它就好
  - Output: 
    
    Code:Canny边缘检测完整实现(代码略，见仓库)

    我已经为你实现了完整的Canny边缘检测算法！这个实现包括：

    主要功能
    - canny_edge_detection() - 主函数
    完整实现Canny的5个步骤
    可调节参数：低阈值、高阈值、高斯sigma
    - create_gaussian_kernel() - 生成高斯卷积核
    默认5×5大小，sigma=1.4
    - non_maximum_suppression() - 非极大值抑制
    将边缘细化到单像素宽度
    根据梯度方向智能比较邻域
    - double_threshold_and_hysteresis() - 双阈值和边缘连接
    分类强边缘和弱边缘
    保留与强边缘连接的弱边缘
    - demo_canny() - 演示函数
    可视化各个步骤的结果
    对比不同参数的效果