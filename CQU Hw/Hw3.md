# 机器学习第三章作业

3.1、在什么情形下式子不必考虑偏置项b

1. 中心化数据：在数据预处理过程中对输入数据进行中心化处理，使其均值为0。
2. 激活函数包含偏置：在神经网络的某些层中，激活函数本身已经包含了偏置项。

3.3、编程实现对率回归，并给出西瓜数据集3.0α上的结果

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, beta):
    samples = X.shape[0]
    X_with_intercept = np.hstack([X, np.ones((samples, 1))])
    predictions = np.dot(X_with_intercept, beta)
    cost = -np.mean(y * predictions - np.log(1 + np.exp(predictions)))
    return cost

def compute_gradient(X, y, beta):
    samples = X.shape[0]
    X_with_intercept = np.hstack([X, np.ones((samples, 1))])
    predictions = sigmoid(np.dot(X_with_intercept, beta))
    error = predictions - y
    gradient = np.dot(X_with_intercept.T, error) / samples
    return gradient

def gradient_descent(X, y, beta, learning_rate, num_iterations, print_cost=False):
    for i in range(num_iterations):
        grad = compute_gradient(X, y, beta)
        beta = beta - learning_rate * grad
        
        if print_cost and i % 100 == 0:
            print(f"{i}th iteration, cost: {compute_cost(X, y, beta)}")
            
    return beta

def initialize_parameters(dim):
    return np.zeros((dim, 1))

def logistic_regression(X, y, num_iterations=1000, learning_rate=0.01, print_cost=False):
    n_features = X.shape[1]
    beta = initialize_parameters(n_features + 1)  # +1 for the intercept
    beta = gradient_descent(X, y, beta, learning_rate, num_iterations, print_cost)
    
    return beta

if __name__ == '__main__':
    data_path = 'watermelon3_0_Ch.csv'
    data = pd.read_csv(data_path).values

    is_good = data[:, 9] == '是'
    is_bad = data[:, 9] == '否'

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    plt.scatter(data[:, 7][is_good], data[:, 8][is_good], c='k', marker='o')
    plt.scatter(data[:, 7][is_bad], data[:, 8][is_bad], c='r', marker='x')

    plt.xlabel('密度')
    plt.ylabel('含糖量')

    beta = logistic_model(X, y, print_cost=True, method='gradDesc', learning_rate=0.3, num_iterations=1000)
    w1, w2, intercept = beta
    x1 = np.linspace(0, 1)
    y1 = -(w1 * x1 + intercept) / w2

    ax1, = plt.plot(x1, y1, label=r'my_logistic_gradDesc')

    lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)
    lr.fit(X, y)

    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(J_cost(X, y, lr_beta))
    w1_sk, w2_sk = lr.coef_[0, :]

    x2 = np.linspace(0, 1)
    y2 = -(w1_sk * x2 + lr.intercept_) / w2

    ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')

    plt.legend(loc='upper right')
    plt.show()
```

3.5、编辑实现线性判别分析，并给出西瓜数据集 3.0α 上的结果.

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class LDA:
    def fit(self, X, y, plot=False):
        pos_mask = y == 1
        neg_mask = y == 0
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]

        mean_pos = X_pos.mean(axis=0, keepdims=True)
        mean_neg = X_neg.mean(axis=0, keepdims=True)

        sw = np.dot((X_pos - mean_pos).T, (X_pos - mean_pos)) + np.dot((X_neg - mean_neg).T, (X_neg - mean_neg))
        
        self.w = np.linalg.solve(sw, (mean_pos - mean_neg).T).ravel()

        if plot:
            self._plot_decision_boundary(X_pos, X_neg, mean_pos, mean_neg)

        return self

    def predict(self, X):
        projection = np.dot(X, self.w)
        threshold = (np.dot(self.w, self.u0.T) + np.dot(self.w, self.u1.T)) / 2
        return (projection >= threshold).astype(int)

    def _plot_decision_boundary(self, X_pos, X_neg, mean_pos, mean_neg):
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pos[:, 0], X_pos[:, 1], c='k', marker='o', label='Positive')
        plt.scatter(X_neg[:, 0], X_neg[:, 1], c='r', marker='x', label='Negative')

        x_vals = np.linspace(X[:, :, 0].min(), X[:, :, 0].max(), 100)
        y_vals = -(self.w[0] / self.w[1]) * x_vals
        plt.plot(x_vals, y_vals, label='Decision Boundary')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data_path = 'path/to/watermelon_dataset.csv'
    data = pd.read_csv(data_path).values

    X = data[:, 7:9].astype(float)  
    y = np.where(data[:, 9] == '是', 1, 0) 

    lda = LDA()
    lda.fit(X, y, plot=True)
    predictions = lda.predict(X)
    print(predictions)
    print(y)
```
