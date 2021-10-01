import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def computer_alpha(X, y):
    """Computes alpha."""
    
    numSamples = len(X)

    P = matrix(np.outer(y,y) * np.inner(X,X))
    q = matrix(np.ones(numSamples) * -1)

    G = matrix(np.diag(np.ones(numSamples) * -1))
    h = matrix(np.zeros(numSamples))

    A = matrix(y, (1,numSamples), 'd')
    b = matrix(0.0)

    solvers.options['show_progress']=False
    solution = solvers.qp(P, q, G, h, A, b)
    alpha=solution['x']
    
    return np.array(alpha)

def fit_and_predict(X, y, test):
    """Fits the SVM model and performs classification.
    
    Args:
        X: array data set.
        y: labels.
        test: list test data point.
    Return:
        value predict label for test.
    """

    alp = computer_alpha(X, y)
    tmp = np.zeros((len(X),2))
    for i in range(len(X)):
        tmp[i,:] = alp[i]*y[i]*X[i,:]
    w = np.sum(tmp, axis=0)
    sv_idx = (alp > 1e-4).flatten()
    b_all = y[sv_idx] - np.dot(X[sv_idx], w)
    b = b_all[0]
    
    y_pre = np.sign(np.dot(test, w) + b)
    return y_pre

if __name__ == "__main__":
    
    # we create 40 separable points
    X, y = make_blobs(n_samples=40, centers=2, random_state=6)
    a = fit_and_predict(X, y, [8,-4])
    print(a)
    plt.scatter(X[:,0], X[:,1], c=y)

