import math
import random

def ThresholdTreeConstruction(X, C, delta):
    # Create a  tree T with a root node named r
    T = {'r': {'centers': C, 'left': None, 'right': None}}
    t = 1

    # While Tt contains a leaf with at least two distinct centers do:
    while True:
        leaves_to_split = []
        for u in T[-1]:
            # Find leaves with more than one center
            if T[-1][u]['left'] is None and T[-1][u]['right'] is None and len(T[-1][u]['centers']) > 1:
                leaves_to_split.append(u)
        if not leaves_to_split:
            break
        # Sample it, θt, and σt uniformly at random
        i = random.randint(1, len(X[0]))
        # Sample sigma randomly as either 0 or 1
        theta = random.uniform(0, 1)
        #Sample sigma randomly as either -1 or 1
        sigma = random.choice([-1, 1])
        # Set the value of epsilon
        epsilon = min(delta / (15 * math.log(len(C))), 1/384)
        # For each leaf u in the tree Tt containing more than one center, split node u using Divide-and-Share
        new_tree = dict(T[-1])
        for u in leaves_to_split:
            new_tree[u] = DivideAndShare(u, i, theta, sigma, epsilon, new_tree[u])
        T.append(new_tree)

    # Update t = t + 1
    t += 1

    return T