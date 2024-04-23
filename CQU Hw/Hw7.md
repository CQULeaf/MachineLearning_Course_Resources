# 机器学习第七章作业

7.1 试使用极大似然法估算回瓜数据集 $3.0$ 中前 $3$ 个属性的类条件概率

以第一个属性色泽为例，其值计数如下：

|好瓜\色泽 | 乌黑 |浅白 |青绿|
|---|---|---|---|
|否|2|4|3|
|是 |4 |1 |3|

令 $ p_{乌黑|是}=p_{1} $ 表示好瓜中色泽为“乌黑”的概率，$ p_{浅白|是}=p_{2} $ 为好瓜中“浅白”的概率，
$ p_{青绿|是}=p_{3} $ ， $ p_{3} = 1-p_{2}-p_{3} $ ， $ D_{是} $ 表示好瓜的样本，其他类同，
于是色泽属性的似然概率则可表示为$ L(p)=P(X_{色泽}|Y=是)=\prod_{x\in D_{是}}P(x)=p_{1}^{4}p_{2}^{1}(1-p_1-p_{2})^{3} $ ,
其对数似然为：$ LL(p)=4ln(p_1)+ln(p_2)+3ln(1-p_1-p_2) $ ，分别对 $ p_1,p_2 $ 求偏导并使其为零，
即可得 $ p_1,p_2 $ 的极大似然估计： $ \hat{p_1}=\frac{1}{2},\hat{p_2}=\frac{1}{8} $,
$ \hat p_{3} =1-\hat p_{2}-\hat p_{3}=\frac{3}{8} $ ，同理可得  $ P(X_{色泽}|Y=否) $ 的似然概率，
进而求得类为“否”时各取值条件概率的极大似然估计。

其他两个属性同理。

7.3 试编程实现拉普拉斯修正的朴素贝叶斯分类器，并以西瓜数据集 $3.0$ 为训练集，对 $p.151$ "测1" 样本进行判别

```python
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

class NaiveBayesClassifier:
    """朴素贝叶斯分类器，适用于处理包含离散和连续特征的数据集。"""
    
    def __init__(self):
        self.prob_class1 = None
        self.features_stats_class1 = []
        self.features_stats_class0 = []

    def fit(self, X, y):
        """根据给定的特征X和目标y，训练朴素贝叶斯模型。"""
        m, n = X.shape
        self.prob_class1 = (np.sum(y == '是') + 1) / (m + 2)  # 拉普拉斯平滑
        
        X_class1 = X[y == '是']
        X_class0 = X[y == '否']

        for i in range(n):
            feature = X.iloc[:, i]
            feature_class1 = X_class1.iloc[:, i]
            feature_class0 = X_class0.iloc[:, i]
            is_continuous = type_of_target(feature) == 'continuous'
            
            if is_continuous:
                # 连续特征: 计算均值和方差
                stats_class1 = (feature_class1.mean(), feature_class1.var())
                stats_class0 = (feature_class0.mean(), feature_class0.var())
            else:
                # 离散特征: 计算概率
                unique_values = feature.unique()
                probs_class1 = np.log((pd.value_counts(feature_class1, sort=False).reindex(unique_values, fill_value=0) + 1) / (len(feature_class1) + len(unique_values)))
                probs_class0 = np.log((pd.value_counts(feature_class0, sort=False).reindex(unique_values, fill_value=0) + 1) / (len(feature_class0) + len(unique_values)))
                stats_class1 = probs_class1
                stats_class0 = probs_class0
            
            self.features_stats_class1.append((is_continuous, stats_class1))
            self.features_stats_class0.append((is_continuous, stats_class0))

    def predict(self, X):
        """预测给定特征集X的类别。"""
        log_prob1 = np.log(self.prob_class1)
        log_prob0 = np.log(1 - self.prob_class1)
        
        for i, feature_stats in enumerate(zip(self.features_stats_class1, self.features_stats_class0)):
            feature_value = X[i]
            is_continuous, stats_class1 = feature_stats[0]
            _, stats_class0 = feature_stats[1]
            
            if is_continuous:
                mean1, var1 = stats_class1
                mean0, var0 = stats_class0
                log_prob1 += -0.5 * np.log(2 * np.pi * var1) - ((feature_value - mean1) ** 2) / (2 * var1)
                log_prob0 += -0.5 * np.log(2 * np.pi * var0) - ((feature_value - mean0) ** 2) / (2 * var0)
            else:
                log_prob1 += stats_class1.get(feature_value, -np.inf)  # 使用get防止键不存在
                log_prob0 += stats_class0.get(feature_value, -np.inf)
        
        return '是' if log_prob1 > log_prob0 else '否'

# 使用示例
if __name__ == '__main__':
    data_path = r'watermelon3_0_Ch.csv'
    data = pd.read_csv(data_path, index_col=0)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    classifier = NaiveBayesClassifier()
    classifier.fit(X, y)

    x_test = X.iloc[0, :]
    print(classifier.predict(x_test))
```

7.7 给定 d 个二值属性的二分类任务，假设对于任何先验概率项的估算至少需 $30$ 个样例，则在朴素贝叶斯分类器式 (7.15) 中估算先验概率项 $ P(c) $ 需 $30 x 2 = 60$ 个样例.试估计在 AODE 式 (7.23) 中估算先验概率项 $ P(c,x_i) $ 所需的样例数(分别考虑最好和最坏情形) .

这里“假设对于任何先验概率项的估算至少需 $30$ 个样例”意味着在所有样本中， 任意 $ c,x_i $ 的组合至少出现 $30$ 次。

当 $ d=1 $ 时，即只有一个特征 $ x_1 $ ，因为是二值属性，假设取值为 $ 1,0 $ ，那为了估计 $ p(y=1,x_1=1) $ 至少需要 $30$ 个样本，
同样 $ p(y=1,x_1=0) $ 需要额外 30 个样本，另外两种情况同理，所以在 $ d=1 $ 时，最好和最坏情况都需要 120 个样本。

再考虑 $ d=2 $ ，多加个特征 $ x_2 $ 同样取值 $ 1,0 $ ，为了满足求 $ P(c,x_1) $ 已经有了 120 个样本，且 60 个正样本和 60 个负样本；
在最好的情况下，在 60 个正样本中，正好有 30 个样本 $ x_2=1 $ , 30 个 $ x_2=0 $ ，负样本同理，此时这 120 个样本也同样满足计算 $ P(c,x_2) $ 的条件，
所有 $ d=2 $ 时，最好的情况也只需要 120 个样本，$ d=n $ 时同理；在最坏的情况下，120 个样子中， $ x_2 $ 都取相同的值 1 ，
那么为了估算 $ P(c,x_2=0) $ 需要额外 60 个样本，总计 180 个样本，同理计算出 $ d=2,3,4... $ 时的样本数，即每多一个特征，
最坏情况需要多加额外 60 个样本， $ d=n $ 时，需要 $ 60(n+1) $ 个样本。

那么 $ d $个二值属性下，最好情况需要 120 个样本，最好情况需要 $ 60(d+1) $ 个样本。
