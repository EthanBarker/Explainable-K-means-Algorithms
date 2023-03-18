import numpy as np
import pandas as pd
def generate_custom_clusters(n_samples=500, noise=0.1, random_state=None):
    np.random.seed(random_state)
    data = []
    labels = []
    #cluster 1
    center1 = np.array([0, 0])
    points1 = np.random.normal(loc=center1, scale=noise, size=(n_samples // 2, 2))
    data.append(points1)
    labels.extend([0] * (n_samples // 2))
    #cluster 2
    center2 = np.array([1, 1])
    points2 = np.random.normal(loc=center2, scale=noise, size=(n_samples // 4, 2))
    data.append(points2)
    labels.extend([1] * (n_samples // 4))
    #cluster 3
    center3 = np.array([0.5, 0.5])
    points3 = np.random.normal(loc=center3, scale=noise * 2, size=(n_samples // 4, 2))
    data.append(points3)
    labels.extend([2] * (n_samples // 4))
    data = np.concatenate(data)
    labels = np.array(labels)
    return data, labels

X, y = generate_custom_clusters(n_samples=500, noise=0.2, random_state=42)

data = np.column_stack((X, y))
df = pd.DataFrame(data, columns=["Feature_1", "Feature_2", "Label"])
df.to_csv("custom_clusters.csv", index=False)
