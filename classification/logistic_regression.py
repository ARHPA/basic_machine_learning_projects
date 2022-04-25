import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
df = pd.read_csv("ChurnData.csv")

df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
       'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
       'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
       'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
       'lninc', 'custcat', 'churn']]
df["churn"] = df["churn"].astype("int")

X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
       'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
       'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
       'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
       'lninc', 'custcat']])
y = np.asarray(df["churn"])

#preprocessing
from sklearn import preprocessing
norm = preprocessing.StandardScaler().fit(X)
X = norm.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

#create logistic regression
from sklearn.linear_model import LogisticRegression
logre = LogisticRegression(C=0.1, solver='liblinear')
logre.fit(X_train, y_train)

#predict
y_hat = logre.predict(X_test)
y_hat_prob = logre.predict_proba(X_test)

#evaluation
from sklearn.metrics import jaccard_score
print("jaccard index: ", jaccard_score(y_test, y_hat))

from sklearn.metrics import log_loss
print("log los: ", log_loss(y_test, y_hat))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_hat))

from sklearn.metrics import accuracy_score
print("accuracy score: ", accuracy_score(y_test, y_hat))

from sklearn.metrics import f1_score
print("f1 score: ", f1_score(y_test, y_hat))

#confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()

