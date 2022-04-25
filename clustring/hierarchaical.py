import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cars_clus.csv")
df2 = df
#data cleaning
df = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales', 'partition']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index()

#normalization
from sklearn.preprocessing import MinMaxScaler
X = df.values
X = MinMaxScaler().fit_transform(X)

#train data with scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance_matrix
dist = distance_matrix(X, X)
z = linkage(dist, method='average')

from scipy.cluster.hierarchy import fcluster
y = fcluster(z, 5,criterion='maxclust')

from scipy.cluster.hierarchy import dendrogram
from matplotlib import pylab
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (df2['manufact'][id], df2['model'][id], int(float(df2['type'][id])) )
    
dendro = dendrogram(z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
#plt.show()

import matplotlib.cm as cm
n_clusters = max(y)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
#plt.figure(figsize=(16,14))

#for color, label in zip(colors, cluster_labels):
#    subset = df[y == label]
#    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset['horsepow'], subset['mpg'])

plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()