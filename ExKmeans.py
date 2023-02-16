import random
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


class Node:
    def __init__(self, centers, coordinates=None, theta=None, sigma=None, epsilon=None, left=None, right=None):
        self.centers = centers
        self.coordinates = coordinates
        self.theta = theta
        self.sigma = sigma
        self.epsilon = epsilon
        self.left = left
        self.right = right

def divide_and_share(node, coordinate, theta, sigma, epsilon):
    centers = node.centers
    if len(centers) == 1:
        return node

    mean = np.mean(centers, axis=0)
    distances = np.linalg.norm(centers - mean, axis=1)
    R = max(distances)
    t = random.uniform(0, R ** 2)
    threshold = math.sqrt(t)
    left_cut = (1 - epsilon) * threshold
    right_cut = (1 + epsilon) * threshold
    left_centers = []
    right_centers = []
    for center in centers:
        if center[coordinate] < left_cut:
            left_centers.append(center)
        elif center[coordinate] > right_cut:
            right_centers.append(center)
        else:
            left_centers.append(center)
            right_centers.append(center)
    left_node = Node(left_centers)
    right_node = Node(right_centers)
    return left_node, right_node
def threshold_tree_construction(centers, delta):
    k = len(centers)
    tree = Node(centers)
    stack = [tree]
    counter = 0
    max_counter = len(centers)
    while stack and counter < max_counter:
        node = stack.pop()
        if len(node.centers) > 1:
            coordinate = random.randint(0, len(centers[0]) - 1)
            theta = random.choice([0, 1])
            sigma = random.choice([-1, 1])
            epsilon = min(delta / (15 * np.log(k)), 1 / 384)
            left_node, right_node = divide_and_share(node, coordinate, theta, sigma, epsilon)
            node.left = left_node
            node.right = right_node
            stack.append(node.left)
            stack.append(node.right)
            counter += 1
    return tree

def plot_tree(tree):
    print(tree.centers)
    if tree is None:
        return
    if tree.left is None and tree.right is None:
        plt.scatter(tree.centers[:, 0], tree.centers[:, 1], marker='o', color='red')
    else:
        plt.scatter(tree.centers[:, 0], tree.centers[:, 1], marker='o', color='blue')
        plot_tree(tree.left)
        plot_tree(tree.right)

# Start the timer
start_time = time.time()

# load the iris dataset
iris = load_iris()
centers = np.array(iris.data)

# build the threshold tree
delta = 1
ExKmeans = threshold_tree_construction(centers, delta)

print(ExKmeans)
plot_tree(ExKmeans)
#plt.show()

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time, "seconds")
