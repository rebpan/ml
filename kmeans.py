#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def direct_way(x, c):
    """Lloyd k-means algorithms.

    Args:
        x: d*n array dataset.
        c: value clusters number.

    Returns:
        f: n*c array label.
    """

    d, n = x.shape
    random_indices = np.random.choice(n, size=c, replace=False)
    m = x[:, random_indices]

    it = 1000
    while it > 0:
        f = np.zeros((n, c))
        for i in range(n):
            gap = np.tile(x[:, i].reshape(2, 1), (1, 7)) - m
            dist = np.linalg.norm(gap, axis=0)
            f[i, np.argmin(dist)] = 1

        m = np.dot(np.dot(x, f), np.linalg.inv(np.dot(f.T, f)))
        it -= 1

    return f


def draw(f, c):
    """Draw result.

    Args:
        f: n*c label array.
        c: value clusters number.
    """

    for j in range(c):
        cluster = np.where(f[:, j] == 1)[0]
        plt.plot(x[0, cluster], x[1, cluster], 'o')
    plt.show()
    

if __name__ == '__main__':

    data = np.loadtxt('Aggregation.txt')
    x = data[:, 0:2]
    x = x.T
    y = data[:, 2]
    c = 7

    # f = direct_way(x, c)
    d, n = x.shape
    random_indices = np.random.choice(n, size=c, replace=False)
    m = x[:, random_indices]
    f = np.zeros((n, c))
    for i in range(n):
        gap = np.tile(x[:, i].reshape(2, 1), (1, 7)) - m
        dist = np.linalg.norm(gap, axis=0)
        f[i, np.argmin(dist)] = 1

    f_copy = np.empty_like(f)
    f_copy[:] = f
    def objective(s):
        sum = 0
        f_copy[i, k] = s
        for l in range(c):
            sum -= np.dot(np.dot(np.dot(f_copy[:, l].reshape(1, n), x.T), x), f_copy[:, l].reshape(n, 1))
        return sum[0]
        
    it = 10
    while it > 0:
        for i in range(n):
            for k in range(c):
                result = minimize(objective, x0=0)
            q = np.argmin(f_copy[i, :])
            p = np.where(f[i, :] == 1)[0][0]
            if q != p:
                f[:,[p, q]] = f[:, [q, p]]
            f_copy[:] = f

    draw(f, c)
    # https://researchcode.com/code/1714880520/coordinate-descent-algorithms/
