# 机器学习第四章作业

4.1 试证明对于不含冲突数据(即特征向量完全相同但标记不同)的训练集，必存在与训练集一致(即训练误差为 0) 的决策树。

1. 如果一个节点下的所有样本都属于同一类别 \(C\)，那么这个节点成为类别 \(C\) 的叶节点。
2. 如果属性集 \(A\) 为空，或者在当前属性集上所有样本的取值相同，这意味着没有更多的属性可以用来进一步划分样本。在这种情况下，选择当前节点下样本数量最多的类别作为该节点的类别。
3. 如果在某一节点对应的属性值上没有样本（即样本集在该属性值上为空），则选择整个样本集 \(D\) 中最多的类作为该节点的类别标记。

在考虑生成与训练集一致的决策树时，我认为不必考虑第3种情况。因此，决策树的叶节点生成停止的条件是：样本全部属于同一类别，或者已经没有更多的特征可以用来进一步划分样本。由于训练集中没有冲突的数据，所以每个节点的训练误差都是0，意味着在所有的特征用完之前，每个叶节点要么是因为样本完全属于同一类别而停止分裂，要么是因为没有更多的特征可用来分裂。这保证了决策树与训练集的一致性。

4.3 图 4.2 是一个递归算法，若面临巨量数据，则决策树的层数会很深，使用递归方法易导致"栈"溢出。试使用"队列"数据结构，以参数MaxDepth 控制树的最大深度，写出与图 4.2 等价、但不使用递归的决策树生成算法。

```python
class TreeNode:
    def __init__(self, is_leaf=False, label=None, split_attribute=None):
        self.is_leaf = is_leaf  # 是否为叶节点
        self.label = label  # 节点标签
        self.split_attribute = split_attribute  # 分裂属性
        self.children = {}  # 子节点

def TreeGenerate(D, A, maxDepth):
    # 基本情况处理
    if not A or len(set([y for _, y in D])) == 1 or maxDepth == 0:
        label = max(set([y for _, y in D]), key=[y for _, y in D].count)
        return TreeNode(is_leaf=True, label=label)
    
    Node_root = TreeNode()
    # 选择最优划分属性a*，这里需要根据实际情况实现属性选择方法
    a_star = choose_best_attribute(D, A)
    Node_root.split_attribute = a_star
    
    # 对每个可能的属性值a*v，创建子节点
    for a_v in set([x[a_star] for x, _ in D]):
        D_v = [(x, y) for x, y in D if x[a_star] == a_v]
        if not D_v:  # D_v为空
            label = max(set([y for _, y in D]), key=[y for _, y in D].count)
            Node_root.children[a_v] = TreeNode(is_leaf=True, label=label)
        else:
            Node_root.children[a_v] = TreeGenerate(D_v, [a for a in A if a != a_star], maxDepth-1)
    
    return Node_root

def choose_best_attribute(D, A):
    return A[0]

# 示例数据和调用
D = [({'a1': 0, 'a2': 1}, 'yes'), ({'a1': 1, 'a2': 0}, 'no')]
A = ['a1', 'a2']
maxDepth = 2
tree_root = TreeGenerate(D, A, maxDepth)
```

4.9 试将 4.4.2 节对缺失值的处理机制推广到基尼指数的计算中去。

\[ Gini(D) =1-\sum_{k=1}^{\left| y \right|}\tilde{p_{k}}^{2} \] ，属性a的基尼指数可推广为：

\[ Gini\_index(D, a)=p\times Gini\_index(\tilde{D}, a) =p\times\sum_{v=1}^{V}\tilde{v}Gini(D^{v}) \]
