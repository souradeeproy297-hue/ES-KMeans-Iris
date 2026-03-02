# Energy-Efficient Sustainable K-Means (ES-KMeans) Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target

# ES-KMeans parameters
k = 3               # number of clusters
max_iter = 100       # maximum iterations
tol = 0.001          # convergence threshold (adaptive stopping)

# Initialize and fit K-Means (using KMeans++ for smart initialization)
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=max_iter, tol=tol, random_state=42)
kmeans.fit(X)

# Cluster labels
labels = kmeans.labels_

# Optional: compute approximate accuracy by matching clusters to labels
# (For demonstration, may not be exact since K-Means is unsupervised)
accuracy = np.max([accuracy_score(y_true, labels),
                   accuracy_score(y_true, (labels+1)%3),
                   accuracy_score(y_true, (labels+2)%3)]) * 100

print("Clustering Accuracy: {:.2f}%".format(accuracy))
print("Number of Iterations:", kmeans.n_iter_)

# Estimate a simple energy proxy (optional, just for demonstration)
# Here, lower iterations = less energy
energy_proxy = (kmeans.n_iter_ / max_iter) * 100
print("Estimated Energy Proxy: {:.2f}%".format(energy_proxy))
input("Press Enter to exit...")

# Performance-to-Energy Ratio (PER)
PER = accuracy / energy_proxy

print("Performance-to-Energy Ratio (PER): {:.2f}".format(PER))
input("Press Enter to exit...")
