import numpy as np
from sklearn.linear_model import LinearRegression


def correlation(array_1, array_2):
    return ((array_1 - array_1.mean()) * (array_2 - array_2.mean())).mean()/np.sqrt(array_1.std() * array_2.std())

def partial_correlation(X, i, y):
    x = X[:, i]
    Z = np.delete(X, i, 1)
    linear_1 = LinearRegression(fit_intercept=True)
    linear_2 = LinearRegression(fit_intercept=True)
    linear_1.fit(Z, x)
    linear_2.fit(Z, y)
    residuals_1 = x - linear_1.predict(Z)
    residuals_2 = y - linear_2.predict(Z)
    return correlation(residuals_1, residuals_2)