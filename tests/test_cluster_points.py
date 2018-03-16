#!/usr/bin/env python

import sys
sys.path.insert(0, '.')
import numpy as np
import warnings
import pytest

from kmeans_py import kmeans_py

#######################################
#####       HELPER FUNCTIONS      #####
#######################################

def gen_acceptable_data():
    """Helper to generate acceptable data"""
    X = np.array([[1, 2, 3, 4],[9, 8 , 7, 6],[1.5, 2, 3.5, 4]])
    c = np.array([[2, 3, 4, 4],[10, 9, 10, 8]])
    return (X, c)


def gen_unacceptable_data():
    """Helper to generate unacceptable data"""
    X = np.array([[1, 2, 3, 4],[9, 8 , 7, 6],[1.5, 2, 3.5, 4]])
    c = np.array([[2, 3, 4]])
    return (X, c)

#########################################
#############     TESTS     #############
#########################################

def test_initial_values_type():
    """
    Tests that when initial_values attribute is not array, correct error thrown
    """
    data, _ = gen_acceptable_data()
    c = None
    model = kmeans_py.kmeans(data=data, K=2)
    model.initial_values = c
    with pytest.raises(TypeError):
        model.cluster_points()


def test_data_type():
    """
    Tests that when data attribute is not array, correct error thrown
    """
    _, c = gen_acceptable_data()
    data = None
    model = kmeans_py.kmeans(data=data, K=2)
    model.initial_values = c
    with pytest.raises(TypeError):
        model.cluster_points()


def test_cluster_num_type():
    """
    Tests that when cluster number (K) is not integer, correct error thrown
    """
    data, c = gen_acceptable_data()
    model = kmeans_py.kmeans(data=data, K=None)
    model.initial_values = c
    with pytest.raises(TypeError):
        model.cluster_points()


def test_data_and_initial_values_compatible_shape():
    """
    Tests that when data and initial values are incompatible shapes,
    correct error thrown
    """
    data, c = gen_unacceptable_data()
    model = kmeans_py.kmeans(data=data, K=2)
    model.initial_values = c
    with pytest.raises(TypeError):
        model.cluster_points()


def test_cluster_assignment_shape():
    """
    Tests that method returns cluster assignments of correct shape
    """
    data, c = gen_acceptable_data()
    model = kmeans_py.kmeans(data=data, K=2)
    model.initial_values = c
    model.cluster_points()
    assert len(model.cluster_assignments) == data.shape[0]
    assert len(np.unique(model.cluster_assignments)) <= c.shape[0]


def test_toy_example_results():
    """
    Tests that the algorithm correctly clusters toy example
    """
    data, c = gen_acceptable_data()
    model = kmeans_py.kmeans(data=data, K=2)
    model.initial_values = c
    model.cluster_points()
    assert np.array_equal(model.cluster_assignments, np.array([0,1,0]))


def test_failed_to_converge_warning():
    """
    Tests that when algorithm fails to converge, correct warning provided
    """
    data, c = gen_acceptable_data()
    model = kmeans_py.kmeans(data=data, K=2)
    model.initial_values = c

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.cluster_points(max_iter=0)
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "Failed to Converge" in str(w[-1].message)
