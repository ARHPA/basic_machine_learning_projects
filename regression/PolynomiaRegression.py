import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/amirreza/Desktop/basicMachineLearningProjects/regression/FuelConsumption.csv')

msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

plt.scatter(data.ENGINESIZE, data.CO2EMISSIONS, color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
#plt.show()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]]) 

test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]]) 

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
test_x_poly = poly.fit_transform(test_x)

reg.fit(train_x_poly, train_y)
print("coefficients: ", reg.coef_)
print("intercept: ", reg.intercept_)


plt.scatter(data.ENGINESIZE, data.CO2EMISSIONS, color='blue')
XX = np.arange(0, 10, 0.1)
YY = reg.intercept_[0] + reg.coef_[0][1] * XX + reg.coef_[0][2] * pow(XX, 2)

test_y_p = reg.predict(test_x_poly)

from sklearn.metrics import r2_score

print("mean absolute error: %f" % np.mean(np.absolute(test_y - test_y_p)))
print("mean squares error: %f" % np.mean((test_y - test_y_p) ** 2))
print("r2_score: %f" % r2_score(test_y, test_y_p))

plt.plot(XX, YY, 'r') 
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()