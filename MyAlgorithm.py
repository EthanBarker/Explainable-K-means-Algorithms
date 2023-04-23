import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import sys

from sklearn.datasets import load_iris

# Set recurision limit to 5000
sys.setrecursionlimit(5000)

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
        # The dimension used to split the node i = 0 is x and i = 1 is y, i = 2 is diagonal.
        self.i = None

class ThresholdTree:
    """
    A binary threshold tree which is used to partition data points.
    Parameters:
        X: numpy array
            Data points to be clustered.
        C: list
            Centres that belong to the current node.
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
        print("--------------------")
        centers = node.centers
        print(f"Number of centers: {len(centers)}")
        print("--------------------")
        print(f"Centers: {self.X[centers][:, :2]}")
        print("--------------------")
        # Check if a node has already been split or if it only has one centre.
        if node.is_split or len(centers) == 1:
            return None, None
        # Calculate the mean.
        mean = np.mean(self.X[centers][:, :2], axis=0)
        # Find the distance from the furtest centre to the mean.
        R = np.max([np.linalg.norm(self.X[centers[j]] - mean) ** 2 for j in range(len(centers))])
        # Randomly choose a threshold value t.
        t = np.random.choice([0, R])
        if i < 2:
            # Find threshold value for vertical/ horizontal lines.
            threshold = mean[i] - sigma * np.sqrt(theta * t) + epsilon * np.sqrt(theta * R)
        else:
            # Find slope and intercept for diagonal.
            slope = np.random.uniform(-1, 1)
            intercept = mean[1] - slope * mean[0]
            threshold = (slope, intercept)
        # Split the centers into two groups.
        if i < 2:
            left_centers = [c for c in centers if self.X[c, i] <= threshold]
            right_centers = [c for c in centers if self.X[c, i] > threshold]
        else:
            slope, intercept = threshold
            left_centers = [c for c in centers if self.X[c, 1] <= slope * self.X[c, 0] + intercept]
            right_centers = [c for c in centers if self.X[c, 1] > slope * self.X[c, 0] + intercept]
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
        # Set the threshold and i .
        node.threshold = threshold
        node.i = i
        # If both the left/ right child have centers, create child nodes and set is_split to True.
        if len(left_centers) > 0 and len(right_centers) > 0:
            node.left_child = TreeNode(left_centers)
            node.right_child = TreeNode(right_centers)
            node.is_split = True
        # If one child nodes is empty, call divide_and_share.
        else:
            while True:
                left_child, right_child = self.divide_and_share(i, node, theta, sigma, epsilon)
                if left_child is not None and right_child is not None and len(centers) > 1:
                    node.left_child = left_child
                    node.right_child = right_child
                    node.is_split = True
                    self.processed_nodes.add(node)
                    print("--------------------")
                    print(f"Added node with centers {node.centers} to processed nodes.")
                    print("--------------------")
                    break
        # Return child nodes.
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
                    i = np.random.randint(0, 3)
                    # Call divide_and_share to split the node.
                    left_child, right_child = self.divide_and_share(i, node, theta, sigma, epsilon)
                    if left_child is not None and left_child not in self.processed_nodes:
                        # Add the left child to the queue if it has not been processed.
                        queue.append(left_child)
                        print(f"Added node with centers {left_child.centers} as the left child of node with centers {centers}.")
                    if right_child is not None and right_child not in self.processed_nodes:
                        # Add the right child to the queue if it has not been processed.
                        queue.append(right_child)
                        print(f"Added node with centers {right_child.centers} as the right child of node with centers {centers}.")
                    if (left_child is not None and len(left_child.centers) == 1) and (
                            right_child is not None and len(right_child.centers) == 1):
                        # If all nodes have only one center, stop the algorithm.
                        print("--------------------")
                        print(f"Stopping the algorithm because all nodes have only one center.")
                        self.processed_nodes.add(left_child)  # Mark as processed.
                        print(f"Added processed node with centers {left_child.centers}")
                        self.processed_nodes.add(right_child)  # Mark as processed.
                        print(f"Added processed node with centers {right_child.centers}")
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
        # Recursively call plot_clusters on left/ right children.
        plot_clusters(node.left_child, X)
        plot_clusters(node.right_child, X)
        # If node is split and i=0, plot a vertical line.
        if node.is_split and node.i == 0:
            plt.axvline(x=node.threshold, color='k', linestyle='--', linewidth=1)
        # If node is split and i=1, plot a horizontal line.
        elif node.is_split and node.i == 1:
            plt.axhline(y=node.threshold, color='k', linestyle='--', linewidth=1)
        # If node is split and i=2, plot a diagonal line.
        elif node.is_split and node.i == 2:
            slope, intercept = node.threshold
            x_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
            y_values = slope * x_values + intercept
            plt.plot(x_values, y_values, color='k', linestyle='--', linewidth=1)

def convert_centers_to_indices(X, centers):
    C = []
    for c in centers:
        dists = np.linalg.norm(X - c, axis=1)
        index = np.argmin(dists)
        C.append(index)
    C = np.array(C)
    return C
def assign_using_threshold_tree(X, root):
    new_assignments = np.zeros(len(X))
    for i, x in enumerate(X):
        node = root
        while node.left_child is not None:
            if node.i == 2:
                slope, intercept = node.threshold
                y_value = slope * x[0] + intercept
                if x[1] < y_value:
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                if x[node.i] < node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
        new_assignments[i] = node.centers[0]
    return new_assignments

def calculate_cost(centres, X, assignments):
    cost = 0
    for i, cluster in enumerate(centres):
        points_in_cluster = X[assignments == i]
        if len(points_in_cluster) > 0:
            center = np.mean(points_in_cluster, axis=0)
            cluster_cost = np.sum(np.linalg.norm(points_in_cluster - center, axis=1)**2)
            cost += cluster_cost
    return cost

def calculate_cost2(centres, X, assignments):
    cost = 0
    for i, cluster in enumerate(centres):
        points_in_cluster = X[assignments == cluster]
        if len(points_in_cluster) > 0:
            center = np.mean(points_in_cluster, axis=0)
            cluster_cost = np.sum(np.linalg.norm(points_in_cluster - center, axis=1)**2)
            cost += cluster_cost
    return cost

# Start the timer
start_time = time.time()

# load the iris dataset
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

#Run k-means
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init = 10).fit(X)
centers = kmeans.cluster_centers_
assignments = kmeans.predict(X)
print("K-means centers =", centers)
C = convert_centers_to_indices(X, centers)

# Create variables to store the best solution
best_root = None
best_cost = float('inf')
best_new_assignments = None
num_iterations = 1000
sum_costs = 0

# Run the algorithm multiple times
for iteration in range(num_iterations):
    tree = ThresholdTree(X, C, delta=0.1)
    root = tree.build()
    new_assignments = assign_using_threshold_tree(X, root)
    cost2 = calculate_cost2(C, X, new_assignments)

    # Check if the current cost is lower than the best cost
    if cost2 < best_cost:
        best_cost = cost2
        best_root = root
        best_new_assignments = new_assignments

    print(f"Iteration {iteration + 1}: Cost = {cost2}")
    sum_costs += cost2
    average_cost = sum_costs / num_iterations

print(f"Lowest cost: {best_cost}")


# Calculate costs and print info
cost = calculate_cost(centers, X, assignments)
print("K-means Assignments =", assignments)
print("New Assignments =", new_assignments)
print("K-Means Cost =", cost)
print("New Cost =", best_cost)
print("Ratio =", best_cost / cost )
print(f"Average cost after {num_iterations} iterations: {average_cost}")

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time)

visualize_ASCII_tree(best_root)

# Plot the datapoints and threshold lines
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Iris Dataset')
plot_clusters(best_root, X)
plt.show()