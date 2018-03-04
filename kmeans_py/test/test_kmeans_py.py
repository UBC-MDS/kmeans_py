"""User-friendly k-means clustering package"""

# run pytest in bash

import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py
from pathlib import Path


import numpy as np

def test_kmeans_init():
    """
    Testing kmeans_init function
    """
    K = range(0, 10, 1)

    for k in K:
        assert kmeans_py.kmeans_init(data = np.array([], []), K = k) != None


def test_kmeans_plot(test_class):
    '''
    testing the kmeans_plot function

    input: any initialized kmeans class
    tests
        - that there is data to plot
        - that cluster assignments have been made
        - that each data point has been assigned a cluster
        - an image has been saved to file
    '''

    # check to see that there is data to plot
    assert test_class.data != None

    # check to see if cluster assignments have been initialized
    assert test_class.cluster_assignments != None

    # check to see that there is a cluster assigned to each point
    assert len(test_class.cluster_assignments) == test_class.data.shape[0]

    # check to see that an image has been produced
    my_file = Path("/kmeans_plot.png")
    assert my_file.is_file() == True
