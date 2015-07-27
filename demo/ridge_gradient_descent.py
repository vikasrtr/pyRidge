"""
Demo of Ridge Regression (using Gradient Descent)

"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

import sys
sys.path.append('../models')

# include the OLS class
from RidgeGradientDescent import Ridge

data = pd.read_csv('data/machine.data.txt', header=None)

# lets keep 9 attributes
y = data[9]
X = data.drop([0, 1, 9], axis=1)

X = X.values
y = y.values

# pyRidge code
rig = Ridge(num_iters=4000, alpha=0.001, beta=0.05)

rig.fit(X, y)
outs = rig.predict(X)

# scikit-learn code
reg = linear_model.Ridge(alpha=0.05, normalize=True)
reg.fit(X, y)
sklearn_pred = reg.predict(X)

plt.scatter(y, outs, color='g', alpha=0.8, label='pyLinear')
plt.scatter(y, sklearn_pred, color='r', alpha=0.4, label='sklearn')
plt.plot([0, 1200], [0, 1200], 'k-', lw=2)

plt.legend()
plt.show()
