import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py
import numpy as np

#######################################
#####       HELPER FUNCTIONS      #####
#######################################


def gen_acceptable_data():
    """Helper to generate acceptable data"""

    np.random.seed(1234)
    x = np.random.uniform(size=100) + [0, 10] * 50
    y = np.random.normal(loc=5, scale=1, size=100) + [0, 10] * 50
    data = np.array([x, y]).transpose()
    cluster_borders = np.percentile(data, [0, 50, 100], axis=0)

    K = range(0, 3)

    return (data, cluster_borders, K)


def gen_unacceptable_data():
    """Helper to generate unacceptable data"""
    X = np.array([[1, 2, 3, 4],[9, 8 , 7, 6],[1.5, 2, 3.5, 4]])
    c = np.array([[2, 3, 4]])
    return (X, c)

#########################################
#############     TESTS     #############
#########################################


def test_no_data():
    """
    Testing correct error handling if no data in given as input.
    """

    try:
        model = kmeans_py.kmeans(K = 5)
        model.initialize_centers(method='kmeanspp')
    except(TypeError):
        assert True
    else:
        assert False


def test_no_K():
    """
    Testing correct error handling if no K value is given as input.
    """

    data = gen_acceptable_data()[0]

    try:
        model = kmeans_py.kmeans(data = data)
        model.initialize_centers(method='kmeanspp')
    except(TypeError):
        assert True
    else:
        assert False


def test_large_K():
    """
    Testing correct error handling for K value that is too large
    """

    data = gen_acceptable_data()[0]

    try:
        model = kmeans_py.kmeans(data = data, K = data.shape[0] + 1)
        model.initialize_centers(method='kmeanspp')
    except(ValueError):
        assert True
    else:
        assert False


def test_invalid_algorithm():
    """
    Testing correct error handling for invalid algorithm input.
    """

    data = gen_acceptable_data()[0]

    try:
        model = kmeans_py.kmeans(data=data, K=100)
        model.initialize_centers(method='blah')
    except(ValueError):
        assert True
    else:
        assert False


def test_K_zero():
    """
     Testing correct error handling of K with 0 value
    """

    data, cluster_borders, _ = gen_acceptable_data()
    k = 0

    try:
        model = kmeans_py.kmeans(data=data, K=k)
        model.initialize_centers(method='kmeanspp')
    except(ValueError):
        assert True
    else:
        assert False


def test_K_one():
    """
     Test correct output shape with valid K = 1 input
    """

    data, cluster_borders, _ = gen_acceptable_data()
    k = 1

    model = kmeans_py.kmeans(data=data, K=k)
    model.initialize_centers(method='kmeanspp')

    assert model.initial_values.shape[0] == k # number of initial values should be the same as K
    assert model.initial_values.shape[1] == model.data.shape[1] # dimensions should match



def test_logical_output_values():
    """
    Test that returned initialization points are unique with valid input.
    """

    data, cluster_borders, _ = gen_acceptable_data()
    k = 2

    model = kmeans_py.kmeans(data=data, K=k)
    model.initialize_centers(method='kmeanspp')

    assert np.array_equal(model.initial_values, np.unique(model.initial_values, axis=0))


def test_output_shape():
    """
     Test correct output shape with valid input data
    """

    data, cluster_borders, _ = gen_acceptable_data()
    k = 2

    model = kmeans_py.kmeans(data=data, K=k)
    model.initialize_centers(method='kmeanspp')

    assert model.initial_values.shape[0] == k # number of initial values should be the same as K
    assert model.initial_values.shape[1] == model.data.shape[1] # dimensions should match


def test_initialization_values():
    """
     Test that algorithm is giving valid outputs.
    """

    data, cluster_borders, _ = gen_acceptable_data()
    k = 2

    model = kmeans_py.kmeans(data=data, K=k)
    model.initialize_centers(method='kmeanspp')

    assert np.min(model.initial_values[:, 0]) >= cluster_borders[0, 0]
    assert np.min(model.initial_values[:, 0]) <= cluster_borders[1, 0]
    assert np.min(model.initial_values[:, 1]) >= cluster_borders[0, 1]
    assert np.min(model.initial_values[:, 1]) <= cluster_borders[1, 1]

    assert np.max(model.initial_values[:, 0]) >= cluster_borders[1, 0]
    assert np.max(model.initial_values[:, 0]) <= cluster_borders[2, 0]
    assert np.max(model.initial_values[:, 1]) >= cluster_borders[1, 1]
    assert np.max(model.initial_values[:, 1]) <= cluster_borders[2, 1]
