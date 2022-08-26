from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
#height (cm)
X = np.array([[188, 197, 199, 204, 208, 211, 214, 215, 217,217,220, 220, 221, 222, 229, 232, 238, 241]]).T
#weight (kg)
y = np.array([[92, 75, 75, 69, 80, 75, 98, 86, 80, 92, 80, 96, 80, 98, 86, 92, 103, 98]]).T

#building xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

#calculating
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A),b) # điểm tối ưu của bài toán Linear Regression  
print("w = ", w)

w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 255, 2)
y0 = w_0 + w_1*x0
y1 = w_1*203 + w_0
y2 = w_1*213 + w_0
print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )

from sklearn import datasets, linear_model

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)

print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
#Visualize data
plt.plot(X.T, y.T, 'ro')
plt.plot(x0, y0)
plt.axis([180, 255, 60, 120])
plt.ylabel('D')
plt.xlabel('B')
plt.show()

