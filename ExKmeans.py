import random
import math
import pandas as pd
import time
import numpy as np
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
        return node, node

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
    Tree = Node(centers)
    stack = [Tree]
    while stack:
        node = stack.pop()
        if len(node.centers) > 1:
            coordinate = random.randint(0, len(centers[0]) - 1)
            theta = random.choice([0, 1])
            sigma = random.choice([-1, 1])
            epsilon = min(delta / (15 * np.log(k)), 1 / 384)
            node.left, node.right = divide_and_share(node, coordinate, theta, sigma, epsilon)
            stack.append(node.left)
            stack.append(node.right)
    return Tree

# Start the timer
start_time = time.time()

# load the iris dataset
iris = load_iris()
centers = iris.data

# build the threshold tree
delta = 1
ExKmeans = threshold_tree_construction(centers, delta)

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time, "seconds")