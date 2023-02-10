import matplotlib.pyplot as plt
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Start the timer
start_time = time.time()

# Perform k-means clustering using sklearn KMeans
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(dataset.iloc[:, :4])

# Get the cluster labels for each datapoint
labels = kmeans.labels_

# Plot the datapoints with different colors for each cluster
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']
for i in range(3):
    ax.scatter(dataset[labels == i]['sepal_length'], dataset[labels == i]['sepal_width'], dataset[labels == i]['petal_length'], c=colors[i], label=f'Cluster {i+1}')

# Add title, axis labels and legend
ax.set_title("Iris Dataset Clusters")
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")
ax.legend()

# End timer and then display time taken to run in terminal
end_time = time.time()
print("Time elapsed: ", end_time - start_time, "seconds")

# Show the graph
plt.show()

