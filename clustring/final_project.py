import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("final_df.csv")

#clearing data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(df["Gender"])
df["Gender"] = le.transform(df["Gender"])

#normalization
from sklearn.preprocessing import StandardScaler
X = np.asarray(df)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

#modeling
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, init='random', n_init=100).fit(X[:,1:])
lable = km.labels_
X_original = np.asarray(df)
for i in range(10):
    msk = lable == i
    plt.scatter(X_original[msk,3], X_original[msk,4], s=X_original[msk,1]*20 + 10)
plt.show()
