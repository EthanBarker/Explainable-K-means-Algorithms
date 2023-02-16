import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class TreeNode:
    def __init__(self, centers):
        self.centers = centers
        self.left_child = None
        self.right_child = None

class ThresholdTree:
    def __init__(self, X, C, delta):
        self.X = X
        self.C = C
        self.delta = delta
        self.root = TreeNode(C)

    def divide_and_share(self, node, i, theta, sigma, epsilon):
        centers = node.centers
        mean = np.mean(self.X[centers], axis=0)
        R = np.max([np.linalg.norm(self.X[centers[j]] - mean) ** 2 for j in range(len(centers))])
        threshold = mean[i] + sigma * np.sqrt(theta * R) + epsilon * np.sqrt(theta * R)
        left_centers = [c for c in centers if self.X[c, i] <= threshold]
        right_centers = [c for c in centers if self.X[c, i] > threshold]
        if len(left_centers) > 0 and len(right_centers) > 0:
            node.left_child = TreeNode(left_centers)
            node.right_child = TreeNode(right_centers)
            return node.left_child, node.right_child
        else:
            return None, None

    def build(self):
        k = len(self.C)
        epsilon = min(self.delta / (15 * np.log(k)), 1/384)
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            centers = node.centers
            if len(centers) > 1:
                theta = np.random.uniform(0, 1)
                sigma = np.random.choice([-1, 1])
                left_child, right_child = self.divide_and_share(node, 0, theta, sigma, epsilon)
                if left_child is not None:
                    queue.append(left_child)
                if right_child is not None:
                    queue.append(right_child)
        return self.root

def flatten_tree(node, Z, k):
    if node.left_child is None and node.right_child is None:
        return k
    k = flatten_tree(node.left_child, Z, k)
    k = flatten_tree(node.right_child, Z, k)
    Z[k, :2] = [node.left_child.centers[0], node.right_child.centers[0]]
    Z[k, 2] = np.linalg.norm(X[node.left_child.centers[0]] - X[node.right_child.centers[0]])
    Z[k, 3] = len(node.left_child.centers) + len(node.right_child.centers)
    return k + 1

# load the iris dataset
iris = pd.read_csv("iris.csv")
X = iris.iloc[:, :-1].values
C = list(range(X.shape[0]))

# construct the threshold tree
delta = 0.1
tree = ThresholdTree(X, C, delta)
tree.build()

# generate the linkage matrix for the dendrogram
Z = np.zeros((X.shape[0]-1, 4))
flatten_tree(tree.root, Z, 0)

# plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.show()
