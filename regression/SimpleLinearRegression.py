from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("/home/amirreza/Desktop/basicMachineLearningProjects/regression/FuelConsumption.csv")

#plt.scatter(data.FUELCONSUMPTION_COMB, data.CO2EMISSIONS,  color='blue')
#plt.xlabel("FUELCONSUMPTION_COMB")
#plt.ylabel("Emission")
#plt.show()

msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

from sklearn import linear_model

reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
reg.fit(train_x, train_y)

test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
test_y_p = reg.predict(test_x)

from sklearn.metrics import r2_score

print("mean absolute error: %f" % np.mean(np.absolute(test_y_p - test_y)))
print("residual sum of squares: %f" % np.mean((test_y_p - test_y) ** 2))
print("r2-score: %f" % r2_score(test_y, test_y_p))

plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='blue')
plt.plot(train_x, reg.coef_[0][0] * train_x + reg.intercept_, '-r')
plt.show() 