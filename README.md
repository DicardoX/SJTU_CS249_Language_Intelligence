# SJTU_CS249_Language_Intelligence

[[python做语音信号处理](https://www.cnblogs.com/LXP-Never/p/10078200.html#信号加窗%C2%A0)]

-------------

## 1. 清音段、浊音段和过渡段

- **清音段**（辅音段）：能量低、过零率高，波形类似随机噪声
- **浊音段**（元音段）：能量高、过零率低，波形具有周期性特点，**短时平稳性质**
- 过渡段：从辅音段向元音段信号变化之间的部分，信号变化快，语音信号处理中最复杂、最困难的部分
- **注意！** **能量低 + 过零率低的部分为无话段（speakless）**

---------

## 2. 预加重（`pre-emphasis`）

- **目的**：在一个频段范围内，增加高频，降低低频，增加语音信号整体的信噪比
- **公式**：$x_n = x_n - \alpha x_{n-1}, \alpha = 0.97$ ，边界情况$x_{-1} = x_0$
- 对于语音信号来说，语音的低频段能量较大，能量主要分布在低频段，语音的功率谱密度随频率的增高而下降，这样，鉴频器输出就会高频段的输出信噪比明显下降，从而导致高频传输衰弱，使高频传输困难，这对信号的质量会带来很大的影响。因此，**在传输之前把信号的高频部分进行加重，然后接收端再去重**，提高信号传输质量。
- 预加重和去加重的设计思想是保持输出信号不变，有效降低输出噪声，以达到提高输出信噪比的目的

------

## 3. 自相关：较好去除噪声，表示一个信号和延迟$\Delta t$点后该信号本身的相似性

- 自相关函数在分析随机信号时候是非常有用的。我们在信号与系统中学过，通过傅里叶变换可以将一个时域信号转变为频域，这样可以更简单地分析这个信号的频谱。但这有个前提，那就是我们分析的信号是确定信号，即无噪声的信号（sin就是sin，cos就是cos）。而在真正的通信中，我们的传输环境是非常复杂的，充满了噪声。很多时候噪声的分布服从高斯分布（噪声幅度低的概率大，噪声幅度高的概率小）我们称这种噪声叫高斯白噪声（其对应的信道叫AWGN信道）。在一个信号传输中，这种噪声会叠加在信号上，那接收端我们收到的就不是一个确定信号，而是一个随时间变化的信号。即使我们信号发送端始终发送同一个信号，但由于每次叠加的噪声不同，接收端收到的信号也不同，此时我们管这种信号叫随机信号。随机信号直接进行傅里叶变换后，在频域会产生非常多的噪声频带，如果在噪声较大、信号较小的情况下，噪声的频谱甚至会淹没原信号的频谱，从而让我们无法分析。
- 而自相关函数的定义我们都知道，Rx(Δt)=E[x(t)*x(t+Δt)]，我们会发现，**如果同一个信号x(t)进行自相关后，还是自己，而不同的信号进行自相关后，数值会变得很小。**不论Δt取多少，在发送端发出的信号始终不变，那么确定信号经过自相关运算后就保存了下来，而由于噪声每一时刻都不同，自相关后噪声就趋近于0了。然后我们又知道维纳-辛钦定理，自相关函数的傅里叶变换是功率谱，这样我们又一次将时域信号转换到频域进行分析，同时还滤除了噪声，唯一的不同只是原来的确定信号时域纵轴是电压V，现在的功率谱纵轴是功率W，二者成平方关系罢了。
- 短时自相关函数主要应用于**端点检测和基音的提取**，在韵母基因频率整数倍处将出现峰值特性，通常根据除R(0)外的第一峰值来估计基音，而在声母的短时自相关函数中看不到明显的峰值。
- **一般在较宽的窗下计算**
- **自相关的幅度图代表的意义？** 振幅大小（绝对值）与该帧中正在说话的可能性，以及原始音频数据的振幅有关
- 若：短时能量大 + 过零率大，则自相关振幅 < 短时能量振幅，原因是正负抵消（？）

-------------

## 4. 短时能量和短时平均幅度的主要用途：

- 区分浊音和清音段，因为浊音的短时能量𝐸(𝑖)E(i)比清音大很多；
- 区分声母和韵母的分界和无话段和有话段的分界

-------

## 5. 短时平均过零率

​	对于连续语音信号，过零率意味着时域波形通过时间轴，对于离散信号，如果相邻的取样值改变符号，则称为过零。**浊音时具有较低的过零率，而清音时具有较高的过零率。**

**作用**：

- 利用短时平均过零率可以从背景噪声中找出语音信号；
- 可以用于判断寂静无话段与有话段的起点和终止位置；
- **在背景噪声较小的时候，用平均能量识别较为有效，在背景噪声较大的时候，用短时平均过零率识别较为有效**。

------------



