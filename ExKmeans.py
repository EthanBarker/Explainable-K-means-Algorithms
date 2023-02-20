import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import scipy.cluster.hierarchy as shc

class TreeNode:
    def __init__(self, centers):
        self.centers = centers
        self.left_child = None
        self.right_child = None
        self.is_split = False

class ThresholdTree:
    def __init__(self, X, C, delta):
        self.X = X
        self.C = C
        self.delta = delta
        self.root = TreeNode(C)
        self.processed_nodes = set()

    def divide_and_share(self, node, i, theta, sigma, epsilon):
        centers = node.centers
        if node.is_split or len(centers) == 1:
            return None, None
        mean = np.mean(self.X[centers], axis=0)
        R = np.max([np.linalg.norm(self.X[centers[j]] - mean) ** 2 for j in range(len(centers))])
        t = np.random.choice([0, R])
        threshold = mean[i] - sigma * np.sqrt(theta * t) + epsilon * np.sqrt(theta * R)
        left_centers = [c for c in centers if self.X[c, i] <= threshold]
        right_centers = [c for c in centers if self.X[c, i] > threshold]
        print("--------------------")
        print(f"Node centers: {centers}")
        print(f"Mean: {mean}")
        print(f"R: {R}")
        print(f"Threshold: {threshold}")
        if len(left_centers) > 0:
            print(f"Left centers: {left_centers}")
        if len(right_centers) > 0:
            print(f"Right centers: {right_centers}")
        print("--------------------")
        if len(left_centers) > 0 and len(right_centers) > 0:
            node.left_child = TreeNode(left_centers)
            node.right_child = TreeNode(right_centers)
            node.is_split = True
            self.processed_nodes.add(node)
        elif len(left_centers) > 0:
            node.left_child = TreeNode(left_centers)
            node.is_split = True
            self.processed_nodes.add(node)
        elif len(right_centers) > 0:
            node.right_child = TreeNode(right_centers)
            node.is_split = True
            self.processed_nodes.add(node)
        return node.left_child, node.right_child

    def build(self):
        k = len(self.C)
        epsilon = min(self.delta / (15 * np.log(k)), 1 / 384)
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node not in self.processed_nodes:
                centers = node.centers
                if len(centers) > 1:
                    theta = np.random.uniform(0, 1)
                    sigma = np.random.choice([-1, 1])
                    for i in range(self.X.shape[1]):
                        left_child, right_child = self.divide_and_share(node, i, theta, sigma, epsilon)
                        if left_child is not None and left_child not in self.processed_nodes:
                            queue.append(left_child)
                            print(
                                f"Adding node with centers {left_child.centers} as left child of node with centers {centers}")
                        if right_child is not None and right_child not in self.processed_nodes:
                            queue.append(right_child)
                            print(
                                f"Adding node with centers {right_child.centers} as right child of node with centers {centers}")
                        if (left_child is not None and len(left_child.centers) == 1) and (
                                right_child is not None and len(right_child.centers) == 1):
                            print(
                                f"Stopping the algorithm because all nodes have only one center")
                            self.processed_nodes.add(left_child)
                            self.processed_nodes.add(right_child)
                            break
                self.processed_nodes.add(node)
        return self.root

def visualize_ASCII_tree(node, depth=0):
    if node is None:
        return
    print(" " * depth + "└── ", end="")
    print(node.centers)
    visualize_ASCII_tree(node.left_child, depth + 2)
    visualize_ASCII_tree(node.right_child, depth + 2)


# Start the timer
start_time = time.time()

# load the iris dataset
iris = load_iris()
X = iris.data[:, :2] # CHANGE HERE: Use only the first 2 columns

# Initialize the centers as the first k samples in X
k = 3
C = np.arange(k)

# construct the threshold tree
delta = 0
tree = ThresholdTree(X, C, delta)
root = tree.build()

# plot dendrogram
plt.figure(figsize=(10, 7))
plt.title("Threshold Tree Dendrogram")
dend = shc.dendrogram(shc.linkage(X[root.centers], method='ward'))

plt.show()

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time)