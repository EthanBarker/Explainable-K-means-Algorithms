import time
import numpy as np
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
        queue = [(self.root, 0)]
        while queue:
            node, depth = queue.pop(0)
            centers = node.centers
            if node not in self.processed_nodes:
                self.processed_nodes.add(node)
                if len(centers) > 1:
                    theta = np.random.uniform(0, 1)
                    sigma = np.random.choice([-1, 1])
                    for i in range(self.X.shape[1]):
                        left_child, right_child = self.divide_and_share(node, i, theta, sigma, epsilon)
                        if left_child is not None and left_child not in self.processed_nodes:
                            queue.append((left_child, depth + 1))
                        if right_child is not None and right_child not in self.processed_nodes:
                            queue.append((right_child, depth + 1))
                # Print the node information
                print(f"{'-' * depth}{centers}")
        return self.root

def flatten_tree(node, Z, k, X, depth=0):
    if node.left_child is None and node.right_child is None:
        # Print the leaf node information
        print(f"{'-' * depth}{node.centers}")
        return k
    left_k = flatten_tree(node.left_child, Z, k, X, depth + 1)
    right_k = flatten_tree(node.right_child, Z, left_k, X, depth + 1)
    if node.left_child is not None and node.right_child is not None:
        # Print the non-leaf node information
        print(f"{'-' * depth}{node.centers} -> {node.left_child.centers}, {node.right_child.centers}")
        Z[right_k, :2] = [node.left_child.centers[0], node.right_child.centers[0]]
        Z[right_k, 2] = np.linalg.norm(X[node.left_child.centers[0]] - X[node.right_child.centers[0]])
        Z[right_k, 3] = len(node.left_child.centers) + len(node.right_child.centers)
        return right_k + 1
    else:
        print(f"skipping node with centers {node.centers}")
        return right_k

# Start the timer
start_time = time.time()

# Load the iris dataset
iris = load_iris()
X = iris.data[:, :2] # Use only the first 2 columns

# Initialize the centers as the first k samples in X
k = 3
C = np.arange(k)

# construct the threshold tree
delta = 0
# Build the threshold tree
tree = ThresholdTree(X, C, delta)
root = tree.build()

# Stop the timer and print the elapsed time
end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time)