import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# X, y = make_regression(n_samples=100, n_features=1, noise=50)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# size = [5, 10, 12, 14, 18, 30, 33, 55, 65, 80, 100, 150]
# price = [300, 400, 450, 800, 1200, 1400, 2000, 2500, 2800, 3000, 3500, 9000]
# # zz = [47, 75, 81, 83, 55, 16, 72, 14, 31, 57, 55, 89]
# plt.scatter(size, price)
#
# series_dict = {'X': size, 'y': price}
# df = pd.DataFrame(series_dict)
# X = df[['X']]
# y = df[['y']]
#
# # print(df)
#
# scores = []
# colors = ['green', 'purple', 'gold', 'blue', 'black']
# plt.scatter(X, y, c='red')
#
# for count, degree in enumerate([1, 2, 3, 4, 5]):
#     model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
#     model.fit(X, y)
#
#     scores.append(model.score(X, y))
#     plt.plot(X, model.predict(X), color=colors[count], label='degree %d' % degree)
#     print(scores[count])
#
# plt.legend(loc=2)
# plt.show()

# ----------------------------------------------------------------------------------------------------

# y = a + b1x1 + b2X2 + b3X3 + e
# intercept 似乎就是截距, coef 似乎就是 b1, b2, b3
# y = 721.82840238 - 48.31790883X1 - 1.59768164X2 - 13.55110411X3

size = [5, 10, 12, 14, 18, 30, 33, 55, 65, 80, 100, 150]
distance = [50, 20, 70, 100, 200, 150, 30, 50, 70, 35, 40, 20]
zz = [47, 75, 81, 83, 19, 23, 29, 11, 13, 81, 89, 11]
price = [300, 400, 450, 800, 1200, 1400, 2000, 2500, 2800, 3000, 3500, 9000]
series_dict = {'X1': size, 'X2': distance, 'z': zz,  'y': price}
df = pd.DataFrame(series_dict)

# print(df)

X = df[['X1', 'X2', 'z']]
y = df[['y']]

regr = linear_model.LinearRegression().fit(X, y)
# regr.fit(X, y)

print(regr.score(X, y))
print(regr.intercept_)
print(regr.coef_)

print('------------------------------')

predict_x = [[75, 100, 83]]

predict_y = regr.predict(predict_x)
print(predict_y)