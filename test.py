import numpy as np
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform

def gonzalez(X, n_clusters):
    """A simple greedy algorithm.

    Args:
        X: The observations to cluster. ndarray of shape (n_samples, n_features).
        n_clusters: The number of centroids to initialize. int.

    Returns:
        centers: indices of centers picked. list.
        labels: labels[i] is the code or index of the center the i'th observation
          is closest to.
        radii: the distances of each observation to it's center.
        max_radius: the maximum distance of observations to it's center. float.
    """
    dist = squareform(pdist(X))
    # pick the first center
    max_each_row = np.amax(dist, axis=1)
    minmax = np.argmin(max_each_row)

    centers_indices = []
    centers_indices.append(minmax)
    n_clusters -= 1

    A = deepcopy(dist)
    # pick the rest centers with highest distance to centers
    while n_clusters > 0:
        A[centers_indices, :] = np.zeros((len(centers_indices), A.shape[1]))
        distances_to_centers = A[:, centers_indices]
        min_each_row = np.amin(distances_to_centers, axis=1)
        maxmin = np.argmax(min_each_row)
        centers_indices.append(maxmin)
        n_clusters -= 1

    centers = X[centers_indices, :]
    dist_centers = dist[:, centers_indices]
    labels = centers_indices[np.argmin(dist_centers, axis=1)]
    radii = np.amin(dist_centers, axis=1)
    max_radius = np.amax(radii)

    return centers, labels, radii, max_radius


class KCeter:
    """Gonzalez algorithm.

    Attributes:
        n_clusters: The number of clusters to form as well as the number of
          centroids to generate. int.
    """
    def __init__(self, n_clusters):
        """Inits."""
        self.n_clusters = n_clusters

    def fit(self, X):
        """run fnozalez algorithm on the dataset.

        Args:
            X: Training instances to cluster. array-like of shape (n_samples, n_features).

        Return:
            self: Fitted estimator. Object.
        """
        centers, labels, radii, max_radius = gonzalez(X, self.n_clusters)
        self.centers_ = centers
        self.labels_ = labels
        self.radii = radii
        self.max_radius_ = max_radius
        return self


if __name__ == "__main__":
