import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("drug200.csv")

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = df["Drug"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

DrugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
DrugTree.fit(X_train, y_train)
y_hat = DrugTree.predict(X_test)

from sklearn.metrics import accuracy_score
print("accuracy score = ", accuracy_score(y_test, y_hat))
