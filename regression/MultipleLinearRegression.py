import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("FuelConsumption.csv")
#print(data.describe())

sub_data = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
sub_data = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']]
#print(sub_data.head())

msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]


from sklearn import linear_model
reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg .fit(train_x, train_y)
print("coefficients: ", reg.coef_)
print("intersept: ", reg.intercept_)


test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_p = reg.predict(test_x)


from sklearn.metrics import r2_score

print("mean absolut error: %f" % np.mean(np.absolute(test_y - test_y_p)))
print("mean squares error: %f" % np.mean((test_y - test_y_p) ** 2))
print("r2_score: %f" % r2_score(test_y, test_y_p))


