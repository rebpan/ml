import numpy as np
    
if __name__ == '__main__':
    data = np.loadtxt('.\datasets\Aggregation.txt')
    x = data[:, 0:2]
    x = x.T
    y = data[:, 2]
    c = 7

    d, n = x.shape
    random_indices = np.random.choice(n, size=c, replace=False)
    centers = x[:, random_indices]

    f = np.zeros((n, c))
    for i in range(n):
        for j in range(c):
            
