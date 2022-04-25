import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#create dataset
from sklearn.datasets import make_blobs
center_x, center_y = [4, -2, 1, 10], [4, -1, 1, 4]
X1, y1 = make_blobs(n_samples=100, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)

#train data
from sklearn.cluster import AgglomerativeClustering
agglom = AgglomerativeClustering(n_clusters=4, linkage='average')
agglom.fit(X1, y1)
label = agglom.labels_

'''
for i in range(4):
    msk = (label == i)
    plt.scatter(X1[msk, 0], X1[msk, 1])
for i in range(100):
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]))
plt.show()
'''
from scipy.spatial import distance_matrix
dist = distance_matrix(X1, X1)

from scipy.cluster import hierarchy
hir = hierarchy.linkage(dist, method='average')

dendro = hierarchy.dendrogram(hir)
plt.show()