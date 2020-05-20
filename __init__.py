from .correlation import *
from .distance_correlation import *
from .Hoeffding_independence import *

# Callable scoring wrapper for sklearn.feature_selection.SelectKBest
def wrapper(func):
    def helper(X, y):
        results = np.zeros(X.shape[1])
        for i in range(len(results)):
            results[i] = func(X[:, i], y)
        return results
    return helper

