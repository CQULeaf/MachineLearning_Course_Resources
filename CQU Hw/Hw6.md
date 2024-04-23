# 机器学习第六章作业

6.1 试证明样本空间中任意点 $ x $ 到超平面 $ (w,b) $ 的的距离为式 (6.2)

令A点到超平面（点B）的距离为 $ \gamma $ ，于是 $ \bar{BA}=\gamma*\frac{w}{\left| w \right|} $ ( $ \frac{w}{\left| w \right|} $ 是 $ w $ 同向的单位向量， 对于超平面 $ (w,b) $ 其垂直方向即 $ w $ )，对于B点有： $ w^{T}\bar{B} +b=0 $ ，而 $ \bar{B}= \bar{A}-\bar{BA} $ ，
于是 $ w^{T}(\bar{A}-\gamma*\frac{w}{\left| w \right|}) + b = 0 $ ，
可得 $ w^{T}\ast\bar{A}-\gamma\ast\left| w \right|+b=0\Rightarrow\gamma=\frac{w^{T}\bar{A}+b}{\left| w \right|} $ ，
这里的 $ \bar{A} $ 即书中 $ x $，即可得式（6.2）。

6.5 试述高斯核 SVM 与 RBF 神经网络之间的联系

1. **核函数的应用**：高斯核（也称为RBF核）是SVM中用于处理非线性数据的一种常用核函数，同时也是RBF神经网络中使用的函数。这种核函数能够将原始特征空间映射到一个更高维的空间，从而使原本在原始空间中线性不可分的数据在新的特征空间中变得线性可分。

2. **特征映射策略**：两者都采用了特征映射的策略，即通过一个非线性变换将输入数据映射到一个高维空间，在这个高维空间中寻求最优的线性分类面（对于SVM）或决策边界（对于RBF神经网络）。这种映射使得它们能够有效地处理非线性问题。

3. **处理非线性问题的能力**：高斯核SVM和RBF神经网络都是因其强大的非线性学习能力而受到青睐。它们能够通过相似度（或距离）函数捕捉样本之间的非线性关系，使得在复杂的数据集上也能进行有效的分类或回归分析。

6.7 试给出式 (6.52) 的完整 KKT 条件

$$ \alpha_{i}(f(x_{i})-y_{i}-\epsilon-\xi_{i})=0,a_{i}\geq0,f(x_{i})-y_{i}-\epsilon-\xi_{i}\leq0 $$
$$ \hat\alpha_{i}(y_{i}-f(x_{i})-\epsilon-\hat{\xi_{i}})=0,\hat{a_{i}}\geq0,y_{i}-f(x_{i})-\epsilon-\hat{\xi_{i}}\leq0 $$
$$ u_{i}\xi_{i}=0, u_{i}\geq0, -\xi_{i}\leq0 $$
$$ \hat{u_{i}}\hat{\xi_{i}}=0, \hat{u_{i}}\geq0, -\hat{\xi_{i}}\leq0 $$
