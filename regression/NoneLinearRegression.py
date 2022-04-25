from matplotlib import figure
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

data = pd.read_csv("/home/amirreza/Desktop/basicMachineLearningProjects/regression/china_gdp.csv")

#plt.scatter(data["Year"], data["Value"], color='red')
#plt.show()

X, Y = data["Year"].values, data["Value"].values

X = X / max(X)
Y = Y / max(Y)

'''
msk = np.random.rand(len(X)) < 0.8
train_x = X[msk]
train_y = Y[msk]
test_x = X[~msk]
test_y = Y[~msk]
'''


from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

test_list = np.arange(1, 10, 1)

ans = 0
max_k = 0

for k in test_list:
    def sigmoid(x, b, c, d):
        y = k / (1 + b * np.exp(-c * (x - d)))
        return y
    popt, pcov = curve_fit(sigmoid, X, Y)
    test_y_p = sigmoid(X, *popt)
    r2 = r2_score(Y, test_y_p)
    if (r2 > ans):
        ans = r2
        max_k = k

print("max r2_score : %f, %d" % (ans, max_k))    

def sigmoid(x, b, c, d):
    y = 5.1 / (1 + b * np.exp(-c * (x - d)))
    return y

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(X, Y, 'ro', label='data')
plt.plot(X, test_y_p, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

