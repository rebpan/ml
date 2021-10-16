import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

if __name__ == "__main__":
    data = pd.read_csv('iris.txt', header=None, sep=',')
    data.drop([4], axis=1, inplace=True)
    data = data.to_numpy()
    # change to dimension*sample
    data = data.T
    m, n = data.shape
    # subtract off the mean for each dimension
    mn = np.mean(data, axis=1)
    data = data - np.tile(mn, (n, 1)).T
    # calculate the covariance matrix
    covariance = np.dot(data, data.T) / (n-1)
    # find the eigenvectors and eigenvalues
    v, pc = np.linalg.eig(covariance)
    # sort the variances in decreasing order
    rindices = np.argsort(-1*v)
    v = v[rindices]
    pc = pc[:, rindices]
    # project the original data set
    y = np.dot(pc.T, data)

    pca = PCA(n_components=4)
    pca.fit(data.T)
    print(pca.components_)