# 机器学习第五章作业

5.1 试述将线性函数 $ f(x) = w^{T}x $ 用作神经元激活函数的缺陷

使用线性函数作为激活函数时，无论是在隐藏层还是在输出层（无论传递几层），其单元值（在使用激活函数之前）都还是输入 $x$ 的线性组合，
这个时候的神经网络其实等价于逻辑回归（即原书中的对率回归，输出层仍然使用Sigmoid函数）的，若输出层也使用线性函数作为激活函数，那么就等价于线性回归。

5.3 对于图 5.7 中的 $ v_{ih} $ ，试推导出 BP 算法中的更新公式 (5.13)

$ \triangle{v_{ih}} = -\eta\frac{\partial{E_{k}}}{\partial{v_{ih}}} $，因$ v_{ih} $ 只在计算 $ b_{h} $ 时用上，
所以 $ \frac{\partial{E_{k}}}{\partial{v_{ih}}}=\frac{\partial{E_{k}}}{\partial{b_{h}}} \frac{\partial{b_{h}}}{\partial{v_{ih}}} $ ，
其中 $ \frac{\partial{b_{h}}}{\partial{v_{ih}}}=\frac{\partial{b_{h}}}{\partial{a_{h}}} \frac{\partial{a_{h}}}{\partial{v_{ih}}}=\frac{\partial{b_{h}}}{\partial{a_{h}}} x_{i} $，
所以 $ \frac{\partial{E_{k}}}{\partial{v_{ih}}}=\frac{\partial{E_{k}}}{\partial{b_{h}}} \frac{\partial{b_{h}}}{\partial{a_{h}}} x_{i} =-e_{h}x_{i} $，即得原书中5.13式。

5.7 根据式 (5.18)和 (5.19) ，试构造一个能解决异或问题的单层 RBF 神经网络

```python
import numpy as np
import matplotlib.pyplot as plt

class RBFNetwork:
    def __init__(self, input_dim, num_centers, output_dim):
        """
        初始化RBF神经网络。
        :param input_dim: 输入层维度。
        :param num_centers: 隐藏层中心的数量。
        :param output_dim: 输出层维度。
        """
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        
        # 初始化网络参数
        self.centers = np.random.rand(num_centers, input_dim)
        self.weights = np.zeros((num_centers, output_dim))
        self.biases = np.zeros((output_dim,))
        self.beta = np.random.randn(num_centers)
        
    def _rbf_kernel(self, X):
        """
        计算RBF核。
        :param X: 输入数据。
        :return: RBF核输出。
        """
        X_c = np.expand_dims(X, axis=1) - np.expand_dims(self.centers, axis=0)
        return np.exp(-self.beta * np.sum(X_c ** 2, axis=2))
    
    def forward(self, X):
        """
        网络前向传播。
        :param X: 输入数据。
        :return: 网络输出。
        """
        activations = self._rbf_kernel(X)
        output = np.dot(activations, self.weights) + self.biases
        return output
    
    def compute_loss(self, y_pred, y_true):
        """
        计算损失。
        :param y_pred: 预测值。
        :param y_true: 真实值。
        :return: 均方根误差。
        """
        return np.mean((y_pred - y_true) ** 2)
    
    def fit(self, X, y, learning_rate, num_epochs):
        """
        训练模型。
        :param X: 输入数据。
        :param y: 目标数据。
        :param learning_rate: 学习率。
        :param num_epochs: 迭代次数。
        :return: 训练过程中的损失列表。
        """
        costs = []
        for epoch in range(num_epochs):
            # 前向传播
            y_pred = self.forward(X)
            # 计算损失
            cost = self.compute_loss(y_pred, y)
            costs.append(cost)
            
            # 反向传播和参数更新
            self.weights -= learning_rate * np.dot(self._rbf_kernel(X).T, (y_pred - y)) / X.shape[0]
            self.biases -= learning_rate * np.mean(y_pred - y, axis=0)
        
        return costs

if __name__ == "__main__":
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([[1], [1], [0], [0]])

    # 创建并训练RBF网络
    rbf_net = RBFNetwork(input_dim=2, num_centers=8, output_dim=1)
    costs = rbf_net.fit(X, y, learning_rate=0.003, num_epochs=10000)

    # 绘制损失曲线
    plt.plot(costs)
```
