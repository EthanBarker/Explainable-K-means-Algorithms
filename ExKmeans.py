import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import scipy.cluster.hierarchy as shc

class TreeNode:
    """
    A node in the threshold tree.

    Args:
        centers (list): The indices of the data points associated with this node.
    """
    def __init__(self, centers):
        # The indices of the data points associated with this node.
        self.centers = centers
        # The left child of this node.
        self.left_child = None
        # The right child of this node.
        self.right_child = None
        # Whether or not this node has been split.
        self.is_split = False
        # The threshold value used to split this node.
        self.threshold = None
        self.i = None


class ThresholdTree:
    """
    A binary tree used to partition data points using the threshold algorithm.

    Parameters:
        X: numpy array
            The data points to be clustered.
        C: list
            The indices of the data points that belong to the current node.
        delta: float
            The error parameter for the clustering algorithm.

    Attributes:
        X: numpy array
            The data points to be clustered.
        C: list
            The indices of the data points that belong to the current node.
        delta: float
            The error parameter for the clustering algorithm.
        root: TreeNode object
            The root node of the tree.
        processed_nodes: set
            A set of nodes that have already been processed during the tree construction.
    """
    def __init__(self, X, C, delta):
        # The data points to be clustered.
        self.X = X
        # The indices of the data points that belong to the current node.
        self.C = C
        # The error parameter for the clustering algorithm.
        self.delta = delta
        # The root node of the tree.
        self.root = TreeNode(C)
        # A set of nodes that have already been processed during the tree construction.
        self.processed_nodes = set()

    def divide_and_share(self, node, theta, sigma, epsilon):
        # Get the centers from the node.
        centers = node.centers
        # Print the number of centers.
        print(f"Number of centers: {len(centers)}")
        # If the node has already been split or only contains one center, return None for both children
        # since divide and share fails in this case.
        if node.is_split or len(centers) == 1:
            return None, None
        # Calculate the mean of all the centers.
        mean = np.mean(self.X[centers], axis=0)
        # Compute the maximum distance from each center to the mean.
        R = np.max([np.linalg.norm(self.X[centers[j]] - mean) ** 2 for j in range(len(centers))])
        # Randomly choose a threshold value t from {0, R}.
        t = np.random.choice([0, R])
        i = np.random.randint(0, 2)
        # Compute the threshold value.
        if i == 0:
            # Compute the threshold value for a vertical split.
            threshold = mean[i] - sigma * np.sqrt(theta * t) + epsilon * np.sqrt(theta * R)
        else:
            # Compute the threshold value for a horizontal split.
            threshold = mean[i] - sigma * np.sqrt(theta * t) + epsilon * np.sqrt(theta * R)
        # Split the centers into two groups by the threshold.
        left_centers = [c for c in centers if self.X[c, i] <= threshold]
        right_centers = [c for c in centers if self.X[c, i] > threshold]
        # Print the results of the computation.
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
        # Set the threshold and i attributes of the node.
        node.threshold = threshold
        node.i = i
        # If both the left and right child have centers, create two new child nodes and set the is_split attribute of the current node to True.
        if len(left_centers) > 0 and len(right_centers) > 0:
            node.left_child = TreeNode(left_centers)
            node.right_child = TreeNode(right_centers)
            node.is_split = True
            self.processed_nodes.add(node)
        # If one of the child nodes is empty, recursively call divide_and_share until it can be split into two non-empty children.
        else:
            while True:
                # Recursively call divide_and_share.
                left_child, right_child = self.divide_and_share(node, theta, sigma, epsilon)
                if left_child is not None and right_child is not None:
                    node.left_child = left_child
                    node.right_child = right_child
                    node.is_split = True
                    self.processed_nodes.add(node)
                    break
        # Return the left and right child nodes.
        return node.left_child, node.right_child

    def build(self):
        k = len(self.C)
        # Calculate the value of epsilon using delta and the number of clusters k.
        epsilon = min(self.delta / (15 * np.log(k)), 1 / 384)
        # Initialize a queue with the root node.
        queue = [self.root]
        while queue:
            # Remove the first element from the queue and process it.
            node = queue.pop(0)
            # Check if the node has not been processed before.
            if node not in self.processed_nodes:
                # Get the centers of the node.
                centers = node.centers
                # If the node has more than one center.
                if len(centers) > 1:
                    # Generate random values for theta and sigma.
                    theta = np.random.uniform(0, 1)
                    sigma = np.random.choice([-1, 1])
                    i = np.random.randint(self.X.shape[1])
                    # Divide the node into two children using the divide_and_share method.
                    left_child, right_child = self.divide_and_share(node, theta, sigma, epsilon)
                    if left_child is not None and left_child not in self.processed_nodes:
                        # Add the left child to the queue if it has not been processed before.
                        queue.append(left_child)
                        print(
                            f"Added node with centers {left_child.centers} as the left child of node with centers {centers}.")
                    if right_child is not None and right_child not in self.processed_nodes:
                        # Add the right child to the queue if it has not been processed before.
                        queue.append(right_child)
                        print(
                            f"Added node with centers {right_child.centers} as the right child of node with centers {centers}.")
                    if (left_child is not None and len(left_child.centers) == 1) and (
                            right_child is not None and len(right_child.centers) == 1):
                        # If all nodes have only one center, stop the algorithm.
                        print(f"Stopping the algorithm because all nodes have only one center.")
                        self.processed_nodes.add(left_child)  # Mark the left child as processed.
                        self.processed_nodes.add(right_child)  # Mark the right child as processed.
                        break
                # Mark the current node as processed.
                self.processed_nodes.add(node)
        # Return the root node of the tree.
        return self.root

def visualize_ASCII_tree(node, depth=0):
    # Check if current node exists
    if node is None:
        return
    # Print node's centers and indentation
    print(" " * depth + "└── ", end="")
    print(node.centers)
    # Recursively print left and right child nodes with increased depth
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

# load the iris dataset
iris = load_iris()
X = iris.data[:, :2] # CHANGE HERE: Use only the first 2 columns
y = iris.target


# Initialize the centers as the first k samples in X
k = 3
C = np.arange(k)

# construct the threshold tree
delta = 0
tree = ThresholdTree(X, C, delta)
root = tree.build()

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

# plot dendrogram
#plt.figure(figsize=(10, 7))
#plt.title("Threshold Tree Dendrogram")
#dend = shc.dendrogram(shc.linkage(X[root.centers], method='ward'))

# Plot the datapoints and threshold lines
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris Dataset')
plot_clusters(root, X)
plt.show()