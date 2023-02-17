import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
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
        self.processed_nodes = set()

    def divide_and_share(self, node, i, theta, sigma, epsilon):
        centers = node.centers
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
        elif len(left_centers) > 0:
            node.left_child = TreeNode(left_centers)
        elif len(right_centers) > 0:
            node.right_child = TreeNode(right_centers)
        return node.left_child, node.right_child

    def build(self):
        k = len(self.C)
        epsilon = min(self.delta / (15 * np.log(k)), 1 / 384)
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            centers = node.centers
            if node not in self.processed_nodes:
                self.processed_nodes.add(node)
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
                                f"Stopping the algorithm because both nodes have only one center: {left_child.centers} and {right_child.centers}")
                            # Print the root node, its children, and their children
                            print("--------------------")
                            print("Root node:")
                            print(f"Centers: {self.root.centers}")
                            if self.root.left_child is not None:
                                print("Left child:")
                                print(f"Centers: {self.root.left_child.centers}")
                                if self.root.left_child.left_child is not None:
                                    print("Left grandchild:")
                                    print(f"Centers: {self.root.left_child.left_child.centers}")
                                if self.root.left_child.right_child is not None:
                                    print("Right grandchild:")
                                    print(f"Centers: {self.root.left_child.right_child.centers}")
                            if self.root.right_child is not None:
                                print("Right child:")
                                print(f"Centers: {self.root.right_child.centers}")
                                if self.root.right_child.left_child is not None:
                                    print("Left grandchild:")
                                    print(f"Centers: {self.root.right_child.left_child.centers}")
                                if self.root.right_child.right_child is not None:
                                    print("Right grandchild:")
                                    print(f"Centers: {self.root.right_child.right_child.centers}")
                            print("--------------------")
                            return self.root
        return self.root

def flatten_tree(node, Z, k, X):
    if node.left_child is None and node.right_child is None:
        print(f"leaf node with centers {node.centers}")
        return k
    left_k = flatten_tree(node.left_child, Z, k, X)
    right_k = flatten_tree(node.right_child, Z, left_k, X)
    if node.left_child is not None and node.right_child is not None:
        print(f"adding node with centers {node.centers} to linkage matrix at index {right_k}")
        Z[right_k, :2] = [node.left_child.centers[0], node.right_child.centers[0]]
        Z[right_k, 2] = np.linalg.norm(X[node.left_child.centers[0]] - X[node.right_child.centers[0]])
        Z[right_k, 3] = len(node.left_child.centers) + len(node.right_child.centers)
        return right_k + 1
    else:
        print(f"skipping node with centers {node.centers}")
        return right_k

# Start the timer
start_time = time.time()

# load the iris dataset
iris = load_iris()
X = iris.data[:, :2] # CHANGE HERE: Use only the first 2 columns

# Compute the distance matrix
D = pdist(X)

# Convert D to square distance matrix
D_square = squareform(D)

# Remove duplicates
unique_rows = np.unique(D_square, axis=0)

# Convert the square distance matrix back to condensed distance matrix
D = pdist(unique_rows)

# check for duplicate rows
if len(X) != len(np.unique(X, axis=0)):
    print("Duplicate rows found!")
else:
    print("There are no duplicate rows in the matrix")

# Initialize the centers as the first k samples in X
k = 3
C = np.arange(k)

# construct the threshold tree
delta = 0
tree = ThresholdTree(X, C, delta)
root = tree.build()

# Flatten the tree to get the linkage matrix
Z = np.zeros((2 * len(C) - 1, 4))
flatten_tree(root, Z, k, X)

print(Z)

# Plot the dendrogram
#plt.figure(figsize=(10, 5))
#dendrogram(Z, color_threshold=0.7*np.max(Z[:,2]))
#plt.xlabel("Samples")
#plt.ylabel("Distance")
#plt.title("Dendrogram")
#plt.show()


# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time)