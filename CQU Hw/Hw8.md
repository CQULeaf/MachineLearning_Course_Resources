# 机器学习第八章作业

8.3 从网上下载或自己编程实现 AdaBoost，以不剪枝抉策树为基学习器，在西瓜数据集 3.0α 上训练一个 AdaBoost 集成，并与图 8.4进行比较。

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Node:
    """
    决策树的节点类，用于保存决策树的结构。
    """
    def __init__(self):
        self.feature_index = None
        self.split_point = None
        self.depth = None
        self.left_child = None
        self.right_child = None
        self.leaf_class = None


def compute_gini_index(y, weights):
    """
    计算加权基尼指数。
    :param y: 标签数组。
    :param weights: 对应的样本权重。
    :return: 加权基尼指数。
    """
    unique_classes = np.unique(y)
    total_weight = np.sum(weights)
    gini_index = 1.0
    for cls in unique_classes:
        cls_weight = np.sum(weights[y == cls])
        gini_index -= (cls_weight / total_weight) ** 2
    return gini_index


def find_best_split(feature, labels, weights):
    """
    根据基尼指数找到最佳分割点。
    :param feature: 特征数组。
    :param labels: 标签数组。
    :param weights: 样本权重。
    :return: 最小基尼指数和对应的分割点。
    """
    sorted_indices = np.argsort(feature)
    sorted_feature = feature[sorted_indices]
    sorted_labels = labels[sorted_indices]
    sorted_weights = weights[sorted_indices]

    total_weight = np.sum(weights)
    min_gini = float('inf')
    min_point = None

    for i in range(1, len(feature)):
        split_point = (sorted_feature[i - 1] + sorted_feature[i]) / 2
        left_labels = sorted_labels[:i]
        right_labels = sorted_labels[i:]
        left_weights = sorted_weights[:i]
        right_weights = sorted_weights[i:]

        left_gini = compute_gini_index(left_labels, left_weights)
        right_gini = compute_gini_index(right_labels, right_weights)
        weighted_gini = ((np.sum(left_weights) * left_gini) + (np.sum(right_weights) * right_gini)) / total_weight

        if weighted_gini < min_gini:
            min_gini = weighted_gini
            min_point = split_point

    return min_gini, min_point


def choose_feature_to_split(X, y, weights):
    """
    选择最佳分割特征。
    :param X: 特征矩阵。
    :param y: 标签数组。
    :param weights: 样本权重。
    :return: 最佳特征索引和分割点。
    """
    num_features = X.shape[1]
    best_feature, best_point, min_gini = None, None, float('inf')
    for feature_index in range(num_features):
        feature_gini, split_point = find_best_split(X[:, feature_index], y, weights)
        if feature_gini < min_gini:
            min_gini = feature_gini
            best_feature, best_point = feature_index, split_point
    return best_feature, best_point


def build_tree(X, y, weights, depth=0, max_depth=2):
    """
    递归构建决策树。
    :param X: 特征矩阵。
    :param y: 标签数组。
    :param weights: 样本权重。
    :param depth: 当前深度。
    :param max_depth: 最大深度。
    :return: 树的根节点。
    """
    node = Node()
    node.depth = depth

    # 终止条件
    if depth == max_depth or len(y) <= 1:
        weighted_counts = np.bincount(y, weights=weights, minlength=2)
        node.leaf_class = np.argmax(weighted_counts)
        return node

    feature_index, split_point = choose_feature_to_split(X, y, weights)
    node.feature_index = feature_index
    node.split_point = split_point

    left_mask = X[:, feature_index] <= split_point
    right_mask = ~left_mask

    node.left_child = build_tree(X[left_mask], y[left_mask], weights[left_mask], depth + 1, max_depth)
    node.right_child = build_tree(X[right_mask], y[right_mask], weights[right_mask], depth + 1, max_depth)

    return node


def predict_single(node, x):
    """
    使用决策树预测单个样本。
    :param node: 树节点。
    :param x: 样本特征。
    :return: 预测类别。
    """
    if node.leaf_class is not None:
        return node.leaf_class
    if x[node.feature_index] <= node.split_point:
        return predict_single(node.left_child, x)
    else:
        return predict_single(node.right_child, x)


def predict_tree(tree, X):
    """
    使用决策树预测一组样本。
    :param tree: 树的根节点。
    :param X: 特征矩阵。
    :return: 预测结果数组。
    """
    return np.array([predict_single(tree, x) for x in X])

# 主函数代码
if __name__ == "__main__":
    data_path = '../data/watermelon3_0a_Ch.txt'
    data = pd.read_table(data_path, delimiter=' ')
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    y[y == 0] = -1
    trees, a, agg_est = adaBoostTrain(X, y)
    pltAdaBoostDecisionBound(X, y, trees, a)
```

8.5 试编程实现 Bagging，以决策树桩为基学习器，在西瓜数据集 3.0α 上训练一个 Bagging 集戚，井与图 8.6 进行比较

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample

def classify_stump(X, feature_index, threshold, direction):
    """
    对数据进行单层决策树分类。
    :param X: 数据特征矩阵
    :param feature_index: 用于分类的特征列索引
    :param threshold: 分割阈值
    :param direction: 分类方向，可以是 'lt' (小于等于) 或 'gt' (大于)
    :return: 分类结果数组
    """
    predictions = np.ones((X.shape[0], 1))
    if direction == 'lt':
        predictions[X[:, feature_index] <= threshold] = -1
    else:
        predictions[X[:, feature_index] > threshold] = -1
    return predictions

def build_stump(X, y):
    """
    构建最佳单层决策树。
    :param X: 数据特征矩阵
    :param y: 数据标签
    :return: 最佳单层决策树的信息字典
    """
    m, n = X.shape
    best_stump = {}
    min_error = float('inf')

    for feature_index in range(n):
        range_min = X[:, feature_index].min()
        range_max = X[:, feature_index].max()
        step_size = (range_max - range_min) / 20

        for step in range(21):
            threshold = range_min + step * step_size
            for direction in ['lt', 'gt']:
                predictions = classify_stump(X, feature_index, threshold, direction)
                error = np.mean(predictions.flatten() != y)

                if error < min_error:
                    min_error = error
                    best_stump['feature_index'] = feature_index
                    best_stump['threshold'] = threshold
                    best_stump['direction'] = direction

    return best_stump

def stump_bagging(X, y, num_stumps=20):
    """
    使用 Bagging 方法构建多个单层决策树。
    :param X: 数据特征矩阵
    :param y: 数据标签
    :param num_stumps: 生成的单层决策树数量
    :return: 决策树列表
    """
    stumps = []
    for _ in range(num_stumps):
        X_resampled, y_resampled = resample(X, y)
        stumps.append(build_stump(X_resampled, y_resampled))
    return stumps

def predict_stumps(X, stumps):
    """
    使用多个单层决策树进行预测。
    :param X: 数据特征矩阵
    :param stumps: 决策树列表
    :return: 预测结果数组
    """
    predictions = np.zeros((X.shape[0], len(stumps)))
    for i, stump in enumerate(stumps):
        predictions[:, i] = classify_stump(X, stump['feature_index'], stump['threshold'], stump['direction']).flatten()
    return np.sign(np.sum(predictions, axis=1))

def plot_decision_boundary(X, y, stumps):
    """
    绘制决策边界和数据点。
    :param X: 数据特征矩阵
    :param y: 数据标签
    :param stumps: 决策树列表
    """
    plt.figure(figsize=(8, 6))
    grid_x, grid_y = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 500),
                                 np.linspace(X[:, 1].min(), X[:, 1].max(), 500))
    grid_data = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_pred = predict_stumps(grid_data, stumps).reshape(grid_x.shape)

    plt.contourf(grid_x, grid_y, grid_pred, alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Positive')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Negative')
    plt.title('Decision Boundary and Data Points')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data_path = 'watermelon3_0a_Ch.txt'
    data = pd.read_table(data_path, delimiter=' ')

    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    y[y == 0] = -1  # 调整标签以符合 {-1, 1} 需求

    stumps = stump_bagging(X, y, 21)
    print(f"Accuracy: {np.mean(predict_stumps(X, stumps) == y):.2f}")
    plot_decision_boundary(X, y, stumps)
```

8.7 试析随机森林为何比决策树 Bagging 集成的训练速度更快

决策树的生成过程中，最耗时的就是搜寻最优切分属性；随机森林在决策树训练过程中引入了随机属性选择，大大减少了此过程的计算量； 因而随机森林比普通决策树Bagging训练速度要快。
