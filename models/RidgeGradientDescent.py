"""
Ridge Regression

using Gradient Descent
"""

import numpy as np


class Ridge():

    """Ridge Regression using Gradient Descent

    Using beta for lambda to avoid python conflict
    """

    def __init__(self, num_iters=2000, alpha=0.1, beta=0.1):
        self.num_iters = num_iters
        self.alpha = alpha
        self.beta = beta

    def _compute_cost(self, X, y, w, beta):
        """Compute the value of cost function, J.
        Here J is total Least Square Error
        """
        m = X.shape[0]
        J = (1. / (2. * m)) * \
            (np.sum((np.dot(X, w) - y) ** 2.) + beta * np.dot(w.T, w))
            
        return J

    def _gradient_descent(self, X, y, w, num_iters, alpha, beta):
        """Performs Graddient Descent.
        The threshold is set by num_iters, instead of some value in this implementation
        """
        m = X.shape[0]
        # Keep a history of Costs (for visualisation)
        J_all = np.zeros((num_iters, 1))

        # perform gradient descent
        for i in range(num_iters):
            #             print('GD: w: {0}'.format(w.shape))
            J_all[i] = self._compute_cost(X, y, w, beta)

            w = w - (alpha / m) * \
                (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + beta * w)

        return w, J_all

    def fit(self, X, y):
        """Fit the model
        """
        Xn = np.ndarray.copy(X)
        yn = np.ndarray.copy(y)

        # initialise w params for linear model, from w0 to w_num_features
        w = np.zeros((Xn.shape[1] + 1, 1))

        # normalise the X
        self.X_mean = np.mean(Xn, axis=0)
        self.X_std = np.std(Xn, axis=0)
        Xn -= self.X_mean
        self.X_std[self.X_std == 0] = 1
        Xn /= self.X_std
        
        
        self.y_mean = yn.mean(axis=0)
        yn -= self.y_mean

        # add ones for intercept term
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        self.w, self.J_all = self._gradient_descent(
            Xn, yn, w, self.num_iters, self.alpha, self.beta)

    def predict(self, X):
        """Predict values for given X
        """
        Xn = np.ndarray.copy(X)

        Xn -= self.X_mean
        Xn /= self.X_std
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        return Xn.dot(self.w) + self.y_mean
