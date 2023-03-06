import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import scipy.cluster.hierarchy as shc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TreeNode:
    """
    TreeNode represents a singe node inside of the tree
    Args:
        centers (list): The x and y values of a data point in this node.
    """
    def __init__(self, centers):
        # The x and y coordinates of the data points in this node.
        self.centers = centers
        # The left child of the node.
        self.left_child = None
        # The right child of the node.
        self.right_child = None
        # If this node has been split or not.
        self.is_split = False
        # The threshold value.
        self.threshold = None
        # The dimension used to split the node i = 0 is x and i = 1 is y.
        self.i = None


class ThresholdTree:
    """
    A binary threshold tree which is used to partition data points.
    Parameters:
        X: numpy array
            Data points to be clustered.
        C: list
            Data points that belong to the current node.
        delta: float
            The error parameter.
    Attributes:
        root: TreeNode object
            The root node of the tree.
        processed_nodes: set
            Set of nodes which have already been processed.
    """
    def __init__(self, X, C, delta):
        self.X = X
        self.C = C
        self.delta = delta
        self.root = TreeNode(C)
        self.processed_nodes = set()

    def divide_and_share(self, i, node, theta, sigma, epsilon):
        # Get the centers inside of the node.
        centers = node.centers
        print(f"Number of centers: {len(centers)}")
        print(f"Centers: {self.X[centers][:, :2]}")
        # Check if a node has already been split or if it only has one centre.
        if node.is_split or len(centers) == 1:
            return None, None
        # Calculate the mean.
        mean = np.mean(self.X[centers][:, :2], axis=0)
        # Find the distance from the furtest centre to the mean.
        R = np.max([np.linalg.norm(self.X[centers[j]] - mean) ** 2 for j in range(len(centers))])
        # Randomly choose a threshold value t.
        t = np.random.choice([0, R])
        # Compute the threshold value.
        threshold = mean[i] - sigma * np.sqrt(theta * t) + epsilon * np.sqrt(theta * R)
        # Split the centers into two groups by the threshold.
        left_centers = [c for c in centers if self.X[c, i] <= threshold]
        right_centers = [c for c in centers if self.X[c, i] > threshold]
        print("--------------------")
        print(f"Node centers: {centers}")
        print(f"Mean: {mean}")
        print(f"R: {R}")
        print(f"Threshold (dimension {i}): {threshold}")
        if len(left_centers) > 0:
            print(f"Left centers: {left_centers}")
        if len(right_centers) > 0:
            print(f"Right centers: {right_centers}")
        print("--------------------")
        # Set the threshold and i values.
        node.threshold = threshold
        node.i = i
        # If both the left/ right child have centers, create child nodes and set is_split to True.
        if len(left_centers) > 0 and len(right_centers) > 0:
            node.left_child = TreeNode(left_centers)
            node.right_child = TreeNode(right_centers)
            node.is_split = True
            self.processed_nodes.add(node)
        # If one of the child nodes is empty, recursively call divide_and_share.
        else:
            while True:
                left_child, right_child = self.divide_and_share(i, node, theta, sigma, epsilon)
                if left_child is not None and right_child is not None:
                    node.left_child = left_child
                    node.right_child = right_child
                    node.is_split = True
                    self.processed_nodes.add(node)
                    break
        # Return the child nodes.
        return node.left_child, node.right_child

    def build(self):
        k = len(self.C)
        # Calculate the value of epsilon.
        epsilon = min(self.delta / (15 * np.log(k)), 1 / 384)
        # Initialize queue with the root.
        queue = [self.root]
        while queue:
            # Remove the first element and process it.
            node = queue.pop(0)
            # Check if the node has not been processed.
            if node not in self.processed_nodes:
                # Get the centers of the node.
                centers = node.centers
                if len(centers) > 1:
                    # Generate random values for theta and sigma and i.
                    theta = np.random.uniform(0, 1)
                    sigma = np.random.choice([-1, 1])
                    i = np.random.randint(0, 2)
                    # Call divide_and_share to split the node.
                    left_child, right_child = self.divide_and_share(i, node, theta, sigma, epsilon)
                    if left_child is not None and left_child not in self.processed_nodes:
                        # Add the left child to the queue if it has not been processed.
                        queue.append(left_child)
                        print(
                            f"Added node with centers {left_child.centers} as the left child of node with centers {centers}.")
                    if right_child is not None and right_child not in self.processed_nodes:
                        # Add the right child to the queue if it has not been processed.
                        queue.append(right_child)
                        print(
                            f"Added node with centers {right_child.centers} as the right child of node with centers {centers}.")
                    if (left_child is not None and len(left_child.centers) == 1) and (
                            right_child is not None and len(right_child.centers) == 1):
                        # If all nodes have only one center, stop the algorithm.
                        print(f"Stopping the algorithm because all nodes have only one center.")
                        self.processed_nodes.add(left_child)  # Mark as processed.
                        self.processed_nodes.add(right_child)  # Mark as processed.
                        break
                # Mark node as processed.
                self.processed_nodes.add(node)
        # Return the root.
        return self.root

def visualize_ASCII_tree(node, depth=0):
    # Check if node exists.
    if node is None:
        return
    print(" " * depth + "└── ", end="")
    print(node.centers)
    # Recursively print left and right child.
    visualize_ASCII_tree(node.left_child, depth + 2)
    visualize_ASCII_tree(node.right_child, depth + 2)

def plot_clusters(node, X):
    # If node is None, return
    if node is None:
        return
    else:
        # Recursively call plot_clusters on left and right children of node
        plot_clusters(node.left_child, X)
        plot_clusters(node.right_child, X)
        # If node is split and i=0, plot a vertical line at its threshold value
        if node.is_split and node.i == 0:
            plt.axvline(x=node.threshold, color='k', linestyle='--', linewidth=1)
        # If node is split and i=1, plot a horizontal line at its threshold value
        elif node.is_split and node.i == 1:
            plt.axhline(y=node.threshold, color='k', linestyle='--', linewidth=1)

# Start the timer
start_time = time.time()

# Load the Avila dataset.
data = pd.read_csv("avila-tr.txt", header=None)

# Split the data into features and labels.
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert the labels to integers.
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to numpy arrays.
X_train = np.array(X_train)[:, :2]
X_test = np.array(X_test)[:, :2]

#Run k-means
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init = 10).fit(X_train)
centers = kmeans.cluster_centers_
print("K-means centers =", centers)

# convert centers to indices
C = []
for c in centers:
    dists = np.linalg.norm(X_train - c, axis=1)
    index = np.argmin(dists)
    C.append(index)
C = np.array(C)
print(C)

# construct the threshold tree
tree = ThresholdTree(X_train, C, delta=0.1)
root = tree.build()

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

# plot dendrogram
#plt.figure(figsize=(10, 7))
#plt.title("Threshold Tree Dendrogram")
#dend = shc.dendrogram(shc.linkage(X[root.centers], method='ward'))
