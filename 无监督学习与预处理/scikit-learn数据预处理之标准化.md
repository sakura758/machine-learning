# scikit-learn数据标准化

&emsp;&emsp;数据预处理是指在对数据进行分析或建模之前，对原始数据进行清洗、转换和整理的一系列操作步骤。其目的是使数据变得更加适合用于后续的统计分析、数据挖掘或机器学习任务。

&emsp;&emsp;数据预处理的主要目标包括：数据清洗、数据转换、特征工程、数据集拆分、缩放和标准化、处理时间序列数据等



## 数据标准化方法

- StandardScaler: 标准化缩放，把特征缩放为符合**均值为0和方差为1**的高斯分布。
- MinMaxScaler: 把特征值缩放到给定的最小值和最大值之间，默认是缩放到**[0,1]**之间。
- MaxAbsScaler: 把特征值缩放到**[-1,1]**之间。
- RobustScaler: 使用中位数和四分位数，默认使用**第一个四分位数（25%分位数）和第3个四分位数（75%分位数）之间的范围**，保每个特征的统计属性都位于同一范围。
- Normalizer:  Normalizer使用**L1或者L2**范数来缩放数据，**默认值为L2**，特征向量的欧式长度等于1

```python
mglearn.plots.plot_scaling()
```

![output](C:\Users\34629\Desktop\output.png)



## 数据标准化 / 归一化的作用

- 提升模型精度：标准化 / 归一化使不同维度的特征在数值上更具比较性，提高分类器的准确性。
- 提升收敛速度：对于线性模型，数据归一化使梯度下降过程更加平缓，更易正确的收敛到最优解。



### 代码例示：

```
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
import numpy as np
x = np.array([[3., 3, 2., 1],
              [2., 0., 0., 2],
              [0., 1., 1., 3],
              [1., 2., 3, 0]])

stand = StandardScaler().fit_transform(x)
print("StandardScaler:\n mean: {} and std: {}\n{}".format(stand.mean(), stand.std(),stand))
print('-*-' * 30)
print('\n')


min_max_scaler = MinMaxScaler().fit_transform(x)
print("MinMaxScaler:\n {}".format(min_max_scaler))
print('-*-' * 30)
print('\n')

maxabs =  MaxAbsScaler().fit_transform(x)
print(" MaxAbsScaler:\n {}".format(maxabs))
print('-*-' * 30)
print('\n')

robust = RobustScaler().fit_transform(x)
print("RobustScaler:\n {}".format(robust))
print('-*-' * 30)
print('\n')

nor = Normalizer().fit_transform(x)
print("StandardScaler:\n {}".format(nor))
```

![image-20231024113213754](C:\Users\34629\AppData\Roaming\Typora\typora-user-images\image-20231024113213754.png)

![image-20231024113239140](C:\Users\34629\AppData\Roaming\Typora\typora-user-images\image-20231024113239140.png)





## 各个缩放器的应用场景

从每一个缩放器的计算方法决定的应用场景不一致。

**StandardScaler**：采用标准分进行缩放，标准分为我们提供了一种对不同数据集的数据进行比较的办法， 这些不同数据集的均值和标准差设置都各不一样。通过这种方法， 我们可以把这些数值视为来自同一个数据集或者数据分布，从而进行比较。例如(人类身高）因其各自的尺度（米与公斤）而变化的幅度小于另一个组成部分（例如，体重），如果这些特征不缩放。因为一米的高度变化比一公斤的重量变化重要得多，这显然是不正确的。大多数的场景优先考虑StandardScaler。

**MinMaxScaler**：能够指定把数据缩放到指定范围，如果对数据有输出要求，例如要把数据缩放到【10至20】，可以采用MinMaxScaler。 

**MaxAbsScaler**：采用绝对值进行缩放，缩放范围【-1至1】。StandardScaler和MinMaxScaler会破坏数据的稀疏，稀疏数据并不是无用数据。这种情况可以采用MaxAbsScaler。

注意：\*MinMaxScaler和MaxAbsScaler从计算方法看出，两者对于最大值，最小值异常敏感。 也就决定了如果数据的方差过大，也就是数据差异过大，有异常值，异常值过大或者过小，则效果并不好。\*

**RobustScaler**：使用四分位距缩放，那么对于异常值有很好的鲁棒性。如果数据中包含许多离群值，StandardScaler可能效果不佳，RobustScaler能保持数据的离群特征。

**Normalizer：**使用单个样本至单位范数进行缩放**。**如果计划使用点积或任何其他核的二次形式来量化任何一对样本的相似性，*这种情况可以采用*Normalizer。



## 总结

&emsp;&emsp;在分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA技术进行降维的时候，StandardScaler表现更好（避免不同量纲对方差、协方差计算的影响）；

&emsp;&emsp;在不涉及距离度量、协方差、数据不符合正态分布、异常值较少的时候，可使用MinMaxScaler。（eg：图像处理中，将RGB图像转换为灰度图像后将其值限定在 [0, 255] 的范围）；

&emsp;&emsp;在带有的离群值较多的数据时，推荐使用RobustScaler。



参考：

&emsp;&emsp;书籍：《Introduction to Machine Learning with Python》

&emsp;&emsp;其他：[三种数据标准化方法的对比：StandardScaler、MinMaxScaler、RobustScaler_# 归一化处理 scaler = minmaxscaler()-CSDN博客](https://blog.csdn.net/m0_47478595/article/details/106402843)

[scikit-learn数据预处理之特征缩放 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/454711078)

