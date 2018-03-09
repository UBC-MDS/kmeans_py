"""User-friendly k-means clustering package"""

# run pytest in bash

import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py
from pathlib import Path

import numpy as np
import warnings

def test_kmeans_init():
    """
    Testing kmeans initialize_centers method
    """
    # data = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [1.5, 2, 3.5, 4]])
    # K = range(0, 3, 1)

    # generating random data
    np.random.seed(1234)
    x = np.random.uniform(size=100) + [0, 10] * 50
    y = np.random.normal(loc=5, scale=1, size=100) + [0, 10] * 50
    data = np.array([x, y]).transpose()
    cluster_borders = np.percentile(data, [0, 50, 100], axis=0)

    K = range(0, 3)

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
        model = kmeans_py.kmeans(data = data, K = data.shape[0] + 1)
        model.initialize_centers(algorithm='kmeans++')
    except(ValueError):
        assert True
    else:
        assert False

    # check if valid algorithm has been chosen
    try:
        model = kmeans_py.kmeans(data=data, K=100)
        model.initialize_centers(algorithm='blah')
    except(ValueError):
        assert True
    else:
        assert False

    for k in K:
        model = kmeans_py.kmeans(data = data, K = k)
        # print(model.data)
        # print(model.K)
        model.initialize_centers(algorithm = 'kmeans++')
        assert model.initial_values is not None  #should return something
        assert type(model.initial_values) is np.ndarray # should return array

        len(np.array([]).shape)

        # return empty array if no values should be initialized
        if k == 0:
            assert model.initial_values.shape[0] == 1 # should return one row
            assert model.initial_values.shape[1] == 0  # should return zero columns

        if k > 0:
            assert np.array_equal(model.initial_values, np.unique(model.initial_values, axis=0)) # check that values are unique
            print(model.initial_values)
            assert model.initial_values.shape[0] == k # number of initial values should be the same as K
            assert model.initial_values.shape[1] == model.data.shape[1] # dimensions should match

            if k == 2:
                # check if initialization values fall within the logical clusters
                assert np.min(model.initial_values[:, 0]) >= cluster_borders[0, 0]
                assert np.min(model.initial_values[:, 0]) <= cluster_borders[1, 0]
                assert np.min(model.initial_values[:, 1]) >= cluster_borders[0, 1]
                assert np.min(model.initial_values[:, 1]) <= cluster_borders[1, 1]

                assert np.max(model.initial_values[:, 0]) >= cluster_borders[1, 0]
                assert np.max(model.initial_values[:, 0]) <= cluster_borders[2, 0]
                assert np.max(model.initial_values[:, 1]) >= cluster_borders[1, 1]
                assert np.max(model.initial_values[:, 1]) <= cluster_borders[2, 1]




def test_kmeans_cluster():
    """
    Testing kmeans cluster_points method

    Tests:
     - That the cluster assignment attribute is of correct dimension
     - That it correctly clusters the toy example provided
     - That it gracefully fails when initial values are missing
     - That it gracefully fails when dimension of intial values do not match
       those of the dataset to cluster
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
    assert np.array_equal(model.cluster_assignments, np.array([0,1,0]))

    # test that when the initial values are missing, it throws correct error
    bad_cent = np.array([[2, 3, 4]])
    bad_model = kmeans_py.kmeans(data=X, K=bad_cent.shape[0])
    bad_model.initial_values = None


    try:
        bad_model.cluster_points()
    except(TypeError):
        assert True
    else:
        assert False

    # test when initial value and data are incomp. shape, throws correct error
    bad_model = kmeans_py.kmeans(data=X, K=bad_cent.shape[0])
    bad_model.initial_values = bad_cent

    try:
        bad_model.cluster_points()
    except(TypeError):
        assert True
    else:
        assert False

    # test for correct error when no data was provided
    no_data_model = kmeans_py.kmeans(data=None, K=bad_cent.shape[0])

    try:
        no_data_model.cluster_points()
    except(TypeError):
        assert True
    else:
        assert False

    # test for correct error when no number of clusters is provided
    no_K_model = kmeans_py.kmeans(data=X, K=None)

    try:
        no_K_model.cluster_points()
    except(TypeError):
        assert True
    else:
        assert False

    # test that warning is produced when clustering failed to converge
    model = kmeans_py.kmeans(data=X, K=cent.shape[0])
    model.initial_values = cent

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.cluster_points(max_iter=0)
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "Failed to Converge" in str(w[-1].message)





def test_kmeans_plot():
    '''
    testing the kmeans_plot function

    input: any initialized kmeans class
    tests
        - that input data is present and the correct type
        - that cluster assignments have been made
        - that each data point has been assigned a cluster
        - an image has been saved to file
    '''

    # check to see that data has been initialized
    # check that data is a numpy nd array
    assert test_class.data != None
    assert type(test_class.data) is np.ndarray

    # check to see if cluster assignments have been initialized
    assert test_class.cluster_assignments != None

    # check to see that there is a cluster assigned to each point
    assert len(test_class.cluster_assignments) == test_class.data.shape[0]

    # check to see that an image has been produced
    my_file = Path("/kmeans_plot.png")
    assert my_file.is_file() == True
