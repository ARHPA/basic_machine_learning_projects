import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv("teleCust1000t.csv")

#print(df["custcat"].value_counts())

#df.hist(column='income', bins=100)

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y = df['custcat'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

Ks = 10
acc = np.zeros((Ks - 1))
for k in range(1, Ks): 
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, Y_train)
    yhat = neigh.predict(X_test)
    acc[k - 1] = accuracy_score(Y_test, yhat)

print( "The best accuracy was with",acc.max(), "with k=", acc.argmax()+1) 

plt.plot(range(1, Ks), acc, 'g-')
plt.legend(('Accuracy '))
plt.show()


