import numpy as np
import matplotlib.pylab as plt
import pandas as pd

#
#---
#------
#this code is very messy because i didn't have any experience before this
#------
#---
#

data = pd.read_csv("/home/amirreza/Desktop/basicMachineLearningProjects/regression/final.csv")

msk = list()
for i in range(len(data)):
    if len(data["Area"][i]) > 4:
        msk.append(False)
        data['Area'][i] = 0
    else:
        msk.append(True)

subdata = data

for i in range(len(data)):
    a = data["Area"][i]
    a = int(a)
    data["Area"][i] = a

subdata = data[(data["Area"] != 0)]
subdata = subdata[(subdata["Area"] < 400)]
#subdata = subdata[(subdata["Price"] < 20000000000)]

Area = np.asanyarray(subdata[["Area"]])
Price = np.asanyarray(subdata[["Price"]])
Room = np.asanyarray(subdata[["Room"]])
Parking = np.asanyarray(subdata[["Parking"]])
Elevator = np.asanyarray(subdata[["Elevator"]])
Warehouse = np.asanyarray(subdata[["Warehouse"]])

Area /= (max(Area) - min(Area))
Room = Room / 5

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

reg = linear_model.LinearRegression()
poly = PolynomialFeatures(degree=4)
X = np.concatenate((Area,Room,Elevator,Warehouse,Parking),axis=1)
x_poly = poly.fit_transform(X)
reg.fit(x_poly, Price)
y_hat = reg.predict(x_poly)

from sklearn.metrics import r2_score

print("mean absolute error: %f" % np.mean(np.absolute(Price, y_hat)))
print("r2-score: %f" % r2_score(Price, y_hat))

x = np.arange(0, 1, 0.01)
plt.scatter(Area, Price)
plt.plot(x, reg.intercept_[0] + reg.coef_[0][1] * x - reg.coef_[0][2] * pow(x, 4), 'r')
plt.show()
