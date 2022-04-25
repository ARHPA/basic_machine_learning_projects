import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=2000, centers=[[4,3], [2,-1], [-1,4]], cluster_std=1)

#create model
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=7).fit(X)
label = db.labels_
number_of_labels = len(set(label)) - (1 if -1 in label else 0)

#visualization
for i in range(number_of_labels):
    msk = (label == i)
    plt.scatter(X[msk, 0], X[msk, 1])
msk = (label == -1)
plt.scatter(X[msk, 0], X[msk, 1], color='black')
plt.show()
