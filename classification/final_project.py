import numpy as np
import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv("final.csv")

#create X&y
X = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall']]
X = np.asarray(X)
y = df['output']
y = np.asarray(y)

#normalize
from sklearn.preprocessing import StandardScaler
norm = StandardScaler().fit(X)
X = norm.transform(X)

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

#create SVM

from sklearn import svm
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    Svm = svm.SVC(kernel=kernel)
    Svm.fit(X_train, y_train)

    #predict
    y_hat = Svm.predict(X_test)

    #evaluation
    from sklearn.metrics import classification_report
    print("svm whith %s: " % kernel)
    print(classification_report(y_test, y_hat))

#create decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_hat = dtree.predict(X_test)

from sklearn.metrics import classification_report
print("for decision tree: ")
print(classification_report(y_test, y_hat))


#visualize
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
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[0,1])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
plt.show()
