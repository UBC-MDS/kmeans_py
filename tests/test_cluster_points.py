import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py


import numpy as np
import warnings

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
