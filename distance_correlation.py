from sklearn.metrics import pairwise_distances
import numpy as np

def dCov_n(array_1, array_2):
    n = len(array_1)
    w = np.ones(n)/n
    A = pairwise_distances(array_1)
    x = A.dot(w)
    y = (0.5*x.dot(w))*np.ones(n) - x
    A += np.add.outer(y,y)
    B = pairwise_distances(array_2)
    x = B.dot(w)
    y = (0.5*x.dot(w))*np.ones(n) - x
    B += np.add.outer(y,y)
    d = lambda P, Q: np.sqrt(np.sum(P*Q))
    return d(A,B)/np.sqrt((d(A,A)*d(B,B)))
