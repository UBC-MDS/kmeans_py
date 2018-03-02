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


    # test that no data gives error message
    try:
        model = kmeans_py.kmeans(K = 5)
        model.initialize_centers(algorithm='kmeans++')
    except(TypeError):
        assert True
    else:
        assert False

    # test that no K value gives error message
    try:
        model = kmeans_py.kmeans(data = data)
        model.initialize_centers(algorithm='kmeans++')
    except(TypeError):
        assert True
    else:
        assert False

    # test that K value with too large value gives error message
    try:
        model = kmeans_py.kmeans(data = data, K = 100)
        model.initialize_centers(algorithm='kmeans++')
    except(ValueError):
        assert True
    else:
        assert False

    for k in K:
        model = kmeans_py.kmeans(data = data, K = k)
        model.initialize_centers(algorithm = 'kmeans++')
        assert model.initial_values is not None #should return something
        assert type(model.initial_values) is numpy.ndarray # should return array
        assert model.initial_values == unique(model.initial_values, axis=0) # check that values are unique
        assert model.initial_values.shape[0] == k # number of initial values should be the same as K
        assert model.initial_values.shape[1] == self.data.shape[1] # dimensions should match



def test_kmeans_cluster():
    """
    Testing kmeans cluster_points method
    """

    X = np.array([[1, 2, 3, 4],[9, 8 , 7, 6],[1.5, 2, 3.5, 4]])
    cent = np.array([[2, 3, 4, 4],[10, 9, 10, 8]])

    model = kmeans_py.kmeans(data=X, K=cent.shape[0])
    model.initial_values = cent
    model.cluster_points()

    # test that the the cluster assignments are the right shape
    assert len(model.cluster_assignments) == X.shape[0]
    assert len(np.unique(model.cluster_assignments)) <= cent.shape[0]

    # test that it correctly clustered the toy example
    assert np.allequal(model.cluster_assignments, np.array([0,1,0]))

    # test that when the initial values are missing, it throws correct error
    bad_model = kmeans_py.kmeans(data=X, K=bad_cent.shape[0])
    bad_model.initial_values = None

    try:
        model.cluster_points()
    except(ValueError):
        assert True
    else:
        assert False

    # test when initial value and data are incomp. shape, throws correct error
    bad_cent = np.array([[2, 3, 4]])
    bad_model = kmeans_py.kmeans(data=X, K=bad_cent.shape[0])
    bad_model.initial_values = bad_cent

    try:
        model.cluster_points()
    except(ValueError):
        assert True
    else:
        assert False
