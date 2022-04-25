import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(0)
X, y = make_blobs(n_samples=10000, centers=[[5, 2], [3, 1], [5, 4], [1, 0]], cluster_std=0.5)

from sklearn.cluster import KMeans
k_mean = KMeans(n_clusters=6)
k_mean.fit(X)
k_mean_labels = k_mean.labels_
k_mean_center = k_mean.cluster_centers_

fig = plt.figure()
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_mean_labels))))
ax = fig.add_subplot(111)
for k in range(len(k_mean_center)):
    member = (k_mean_labels == k)
    ax.plot(X[member, 0], X[member, 1], '.')
    ax.plot(k_mean_center[k][0], k_mean_center[k][1], 'o', markerfacecolor='black', markersize=10)
plt.show()
