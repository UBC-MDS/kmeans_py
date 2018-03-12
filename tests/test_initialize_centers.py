import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py


import numpy as np

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
