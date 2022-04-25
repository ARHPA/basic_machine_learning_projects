import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Cust_Segmentation.csv")

#preprocessing
df.drop("Address", axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
X = StandardScaler().fit_transform(X)

#k means
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
k_means.fit(X)
df["lable"] = k_means.labels_

#visualization
fig = plt.figure()
ax = fig.add_subplot(111)
for k in range(len(k_means.cluster_centers_)):
    k_member = (k == k_means.labels_)
    ax.scatter(X[k_member, 3], X[k_member, 2], s=X[k_member, 0] * 10 * np.pi)
plt.show()
