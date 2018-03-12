import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py


import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_kmeans_report():
    '''
    testing the kmeans_report method

    tests
        - that input data is present and the correct type
        - that cluster assignments have been made
        - that each data point has been assigned to a cluster
        - that a cluster summary table has been produced
        - that an assignment summary table has been produced
    '''

    X = np.array([[1, 2, 3, 4],[9, 8 , 7, 6],[1.5, 2, 3.5, 4]])
    model = kmeans_py.kmeans(data = X, K = 2)
    model.initialize_centers()
    model.cluster_points()

    # check to see that input data exists
    assert model.data is not None

    # check that input data is a numpy array
    assert type(model.data) is np.ndarray

    # check to see if cluster assignments exist
    assert model.cluster_assignments is not None

    # check to see that there is a cluster assigned to each point
    assert len(model.cluster_assignments) == model.data.shape[0]

    model.report()

    # check that a summary table has been produced
    assert model.cluster_summary is not None

    # check that summary tables are of type pandas DataFrame
    assert type(model.cluster_summary) is pd.core.frame.DataFrame
    assert type(model.assignment_summary) is pd.core.frame.DataFrame
