from .correlation import *
from .distance_correlation import *
from .Hoeffding_independence import *
from .autocorrelation import *

def wrapper(func):

    """
    Callable scoring wrapper for sklearn.feature_selection.SelectKBest
    """
    
    def helper(X, y):
        results = np.zeros(X.shape[1])
        for i in range(len(results)):
            results[i] = func(X[:, i], y)
        return results
    return helper

