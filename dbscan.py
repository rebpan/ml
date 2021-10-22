import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def range_query(dataset, q, eps):
    """finds neighbors of point q.

    Args:
        dataset: array dataset.
        q: value be queried data point.
        eps: value radius.

    Returns:
        list denote neighbors of q.
    """

    neighbors = []
    for i in range(dataset.shape[0]):
        if np.linalg.norm(dataset[i, :] - dataset[q, :]) < eps:
            neighbors.append(i)
    return neighbors

def dbscan(dataset, eps, min_pts):
    """DBSCAN algorithm.

    Args:
        dataset: array dataset.
        eps: value radius.
        min_pts: value determine core point.

    Returns:
        label list.
    """

    # cluster counter
    c = 0
    # the number of data point
    n = dataset.shape[0]
    pre_label = ['undefined'] * n
    for i in range(n):
        # previously precessed in inner loop
        if pre_label[i] != 'undefined':
            continue
        # find neighbors
        neighbors = range_query(dataset, i, eps)
        # density check
        if len(neighbors) < min_pts:
            # label as noise
            pre_label[i] = None
            continue
        # label initial point
        pre_label[i] = c
        # neighbors to expand
        # seed_set = neighbors.remove(i)
        neighbors.remove(i)
        seed_set = neighbors
        # for q in seed_set:
        #     # change noise to border point
        #     if pre_label[q] == None:
        #         pre_label[q] = c
        #     # previously precessed (e.g., border point)
        #     if pre_label[q] != 'undefined':
        #         continue
        #     # label neighbor
        #     pre_label[q] = c
        #     # find neighbors
        #     neighbors = range_query(dataset, q, eps)
        #     # density check (if q is a core point)
        #     if len(neighbors) > min_pts:
        #         seed_set  = list(set(seed_set + neighbors))
        visited_points = [i]
        while len(seed_set) > 0:
            q = seed_set[0]
            visited_points.append(q)
            if pre_label[q] == None:
                pre_label[q] = c
            # previously precessed (e.g., border point)
            if pre_label[q] != 'undefined':
                seed_set = seed_set[1:]
                continue
            # label neighbor
            pre_label[q] = c
            # find neighbors
            neighbors = range_query(dataset, q, eps)
            # density check (if q is a core point)
            if len(neighbors) > min_pts:
                seed_set  = list(set(seed_set + neighbors))
            to_be_remove = set(seed_set).intersection(set(visited_points))
            seed_set = list(set(seed_set) - to_be_remove)
        # next cluster label
        c += 1
    return pre_label

if __name__ == "__main__":
    dataset_with_label = np.loadtxt('jain.txt')
    dataset = dataset_with_label[:, 0:2]
    # label = dataset_with_label[:, -1]
    pre_label = dbscan(dataset, 2, 3)
    pre_label = np.array(pre_label)
    pre_label_sklearn= DBSCAN(eps=2, min_samples=3).fit_predict(dataset)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # not draw noise
    ax1.scatter(dataset[:, 0], dataset[:, 1], c=pre_label)
    # draw noise
    ax2.scatter(dataset[:, 0], dataset[:, 1], c=pre_label_sklearn)
    plt.show()
