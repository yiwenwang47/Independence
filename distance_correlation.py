from sklearn.metrics import pairwise_distances
import numpy as np

# Calculates distance variance of an array. See dCor_n for details regarding notation.
def primer_dVar(array, reshape_array, w):
    if reshape_array:
        array = array.reshape((-1,1))
    A = pairwise_distances(array)
    x = A.dot(w)
    y = (0.5*x.dot(w))*np.ones(len(array)) - x
    A += np.add.outer(y,y)
    return A

# Distance correlation between two arrays. 
# If the dimension of array_1 is larger than one, set reshape_1=False.
def dCor_n(array_1, array_2, reshape_1=True, reshape_2=True):
    if array_1.std() == 0 or array_2.std() == 0:
        return 0
    n = len(array_1)
    w = np.ones(n)/n
    A = primer_dVar(array_1, reshape_1, w)
    B = primer_dVar(array_2, reshape_2, w)
    d = lambda P, Q: np.sqrt(np.sum(P*Q))
    return d(A,B)/np.sqrt((d(A,A)*d(B,B)))

# Calculates the distance correlation between each feature in X and y.
def dCor_n_wrapped(X, y):
    n, m = X.shape
    results = np.zeros(m)
    w = np.ones(n)/n
    B = primer_dVar(y, True, w)
    d = lambda P, Q: np.sqrt(np.sum(P*Q))
    dVar_y = d(B,B)
    if y.std() != 0:
        return results
    for i in range(m):
        array = X[:, i]
        if array.std() != 0:
            A = primer_dVar(array, True, w)
            results[i] = d(A,B)/np.sqrt((d(A,A)*dVar_y))
    return results
