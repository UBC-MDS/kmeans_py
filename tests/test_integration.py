#!/usr/bin/env python

import sys
sys.path.insert(0, '.')
import numpy as np
import warnings

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

def test_init_and_cluster_integration():
    """
    Tests that intialize_centers and cluster_points methods integrate
    correctly when given valid data
    """
    data, _ = gen_acceptable_data()
    model = kmeans_py.kmeans(data=data, K=2)
    model.initialize_centers()
    model.cluster_points()
    assert np.array_equal(model.cluster_assignments, np.array([0,1,0]))


def test_cluster_and_report_integration():
    """
    Tests that cluster_points and report methods integrate correctly
    when given valid data
    """
    k = 2
    data, c = gen_acceptable_data()
    model = kmeans_py.kmeans(data=data, K=k)
    model.initial_values = c
    model.cluster_points()
    model.report()
    assert model.cluster_summary.shape[0] == k
    assert model.assignment_summary.shape == (data.shape[0], data.shape[1] + 1)


def test_report_type_integration():
    """
    Tests that all methods integrate correctly when given valid data
    """
    k = 2
    data, _ = gen_acceptable_data()
    model = kmeans_py.kmeans(data=data, K=k)
    model.initialize_centers()
    model.cluster_points()
    model.report()
    assert model.cluster_summary.shape[0] == k
    assert model.assignment_summary.shape == (data.shape[0], data.shape[1] + 1)
