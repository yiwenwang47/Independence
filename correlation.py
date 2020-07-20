import numpy as np
from sklearn.linear_model import LinearRegression

# Pearson's correlation between two arrays.
def correlation(array_1: np.ndarray, array_2: np.ndarray) -> np.float:
    if array_1.std() == 0 or array_2.std() == 0:
        return 0
    corr = ((array_1 - array_1.mean()) * (array_2 - array_2.mean())).mean()/(array_1.std() * array_2.std())
    return corr

# Partial correlation test based on Pearson's correlation.
def partial_correlation(X: np.ndarray, i: int, y: np.ndarray) -> np.float:

    """
    Partial correlation test of the ith row of X and y.
    A very naive implementation. Doesn't seem like there is any easily quicker way.
    Could be very time consuming if X has many rows.
    """

    x = X[:, i]
    Z = np.delete(X, i, 1)
    linear_1 = LinearRegression(fit_intercept=True)
    linear_2 = LinearRegression(fit_intercept=True)
    linear_1.fit(Z, x)
    linear_2.fit(Z, y)
    residuals_1 = x - linear_1.predict(Z)
    residuals_2 = y - linear_2.predict(Z)
    return correlation(residuals_1, residuals_2)