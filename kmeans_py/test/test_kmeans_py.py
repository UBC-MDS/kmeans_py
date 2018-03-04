"""User-friendly k-means clustering package"""

# run pytest in bash

import sys
sys.path.insert(0, '.')
from kmeans_py import kmeans_py

import numpy as np

def test_kmeans_init():
    """
    Testing kmeans_init function
    """
    K = range(0, 10, 1)

    for k in K:
        assert kmeans_py.kmeans_init(data = np.array([], []), K = k) != None


def test_kmeans_plot():
    '''
    testing the kmeans_plot function
    '''

    ## initialize class
    ## initialize cluster centers
    ## initialize hard cluster assignments

    assert ## save file somewhere and check that it is there
           ## delete file once it is there
