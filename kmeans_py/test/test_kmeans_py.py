"""User-friendly k-means clustering package"""

# run pytest in bash

import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py

import numpy as np

def test_kmeans_init():
    """
    Testing kmeans initialize_centers method
    """

    data = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [1.5, 2, 3.5, 4]])
    K = range(0, 10, 1)

    for k in K:
        model = kmeans_py.kmeans(data = data, K = k)
        model.initialize_centers(algorithm = 'kmeans++')
        assert model.initial_values != None #should return something
        assert type(model.initial_values) is np.ndarray # should return array
        assert model.initial_values == unique(model.initial_values, axis=0) # check that values are unique



def test_kmeans():
    """
    Testing kmeans cluster_points method
    """

    X = np.array([[1, 2, 3, 4],[9, 8 , 7, 6],[1.5, 2, 3.5, 4]])
    cent = np.array([[2, 3, 4, 4],[10, 9, 10, 8]])

    model = kmeans_py.kmeans(data=X, K=cent.shape[0])
    model.initial_values = cent
    model.cluster_points()

    assert model.cluster_assignments != None
