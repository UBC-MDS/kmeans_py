#!/usr/bin/env python

import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class kmeans():

    def __init__(self, data, K):

        self.data = data
        self.K = K
        self.initial_values = None
        self.cluster_centers = None
        self.cluster_assignments = None
        self.cluster_summary = None
        self.assignment_summary = None

    def initialize_centers(self, method='kmeanspp', seed=None):
        """ Choose Initial K-Means Values

        Arguments
        ---------
        data: array
            the data object that k-means clustering will be applied to.

        K: float
            the number of initial values to be chosen. Should correspond to the number of clusters to be chosen.

        method: string (default = 'kmeanspp')
            the initialisation method specified as a string.

            - 'kmeanspp': K-means++ optimization algorithm. Safer, but more time complex, initialization algorithm compared to Lloyd's algorithm.
            - 'rp': Random point initialization method. Less reliable, but faster than 'kmeanspp' method.

        seed: integer
            the seed to be set if "rp" is specified as method. If None, no seed will be set.

        Returns
        -------
        Array with coordinates for initialization values, where each row is an initialization value and the columns correspond with the columns of the input data object.

        Example
        --------
        import numpy as np

        x = np.random.uniform(0,10,100) + [0, 10]*50
        y = np.random.normal(5,1,100) + [0, 10]*50
        data = np.array([x, y])

        kmeans_init(data, K = 2)

        """
        if self.data is None or type(self.data) is not np.ndarray:
            raise TypeError("Data is missing or of wrong object type. Please specify data as a Numpy array.")

        if self.K is None or self.K%1 != 0:
            raise TypeError("K is missing or is not an integer. Please specify K as an integer.")

        if self.data.shape[0] < self.K:
            raise ValueError("Cannot choose more initialize values than data observations.")

        if self.K == 0:
            raise ValueError("K value cannot be 0. Specify the number of initial values as an integer larger than 0.")

        # format as Numpy array, if data object is not in this format (e.g. nested list)
        if type(self.data) != np.ndarray:
            self.data = np.array(self.data)

        # initialize centroids data object
        # centroids = np.array([])

        # kmeans++ algorithm
        if method == "kmeanspp":
            # use first observation as random first centroid starting point
            centroids = np.array([self.data[0]])

            # assign rest of centroids
            if self.K >= 2:
                # filter through number of centroid assignments (minus 1 that has already been created)
                for count in range(1, self.K):
                    cluster_dist = []

                    # cycle through all data points/possible centroids
                    for point in range(self.data.shape[0]):
                        # determine closest existing centroid to point with squared sum
                        data_row = self.data[point]
                        cluster_dist.append(min(np.sum(np.subtract(data_row, centroids)**2, axis = 1)))

                    # calculate normalizing factor
                    dist_cumsum = sum(cluster_dist)

                    # initialize cdf
                    cluster_dist_cum_probs = []

                    # iterate through data point to centroid minimum distances
                    for cum_count, dist in enumerate(cluster_dist):
                        # create pdf of distances
                        prob = dist/dist_cumsum
                        # initial cdf assigning
                        if cum_count == 0:
                            cluster_dist_cum_probs.append(prob)
                        else:
                            cluster_dist_cum_probs.append(cluster_dist_cum_probs[-1] + prob)

                    # random sample from uniform distribution
                    # we need to stipulate a random point somewhere in the cdf

                    init_samp = np.random.uniform()

                    # centre selected based on cdf
                    # the sample value will have a higher probability of landing on a
                    # generally far away distance (clustered points in different cluster to centroids)
                    # since these have the biggest weight/area in the cdf
                    for cum_count in range(self.data.shape[0]):
                        # assign centroid based on where it is situated on cdf
                        if init_samp < cluster_dist_cum_probs[cum_count]:
                            cent = cum_count
                            break

                    # add centroid
                    centroids = np.vstack((centroids, self.data[cent]))

            self.initial_values = centroids

        # random points method
        elif method == "rp":
            # set seed if specified
            if seed is not None:
                if seed%1 == 0:
                    np.random.seed(seed)
                else:
                    raise ValueError("Invalid seed has been provided. Please specify seed as integer or omit.")

            # select random rows as initialization values
            cent = np.random.randint(0, self.data.shape[0], size=self.K)

            self.initial_values = self.data[cent]

        else:
            raise ValueError("Please specify a valid algorithm to apply.")

        return None

    def cluster_points(self, max_iter=100):
        """
        Perform k-means clustering on the provided data

        Inputs:
         - max_iter: an integer specifying the maximum number of iterations

        Requires:
         - self.data:           provided upon initialization of object
         - self.initial_values: computed by initialize_centers method

        Output:
         - None
         - Called for side effect of updating the cluster_assignments
           and cluster_centers attributes of the kmeans object on
           which this method is called
        """
        # this should be first to avoid Nonetype errors
        if not isinstance(self.initial_values, np.ndarray):
            raise TypeError("Cluster centers is of incorrect type")

        # also check if the data is none
        if not isinstance(self.data, np.ndarray):
            raise TypeError("The data is of incorrect type")

        # make sure there is an integer number of clusters and it is positive
        # this addresses point 7 from feedback
        if not (isinstance(self.K, int) and self.K >= 1):
            raise TypeError("Number of clusters is of incorrect type")

        # dimension of dataset and number of clusters
        n, d = self.data.shape
        k, iv_dim = self.initial_values.shape

        # check that some basic conditions are met
        if d != iv_dim:
            raise TypeError("Initial values and data are not compatible shape")

        # array to hold distance between all points and all centers
        dist_arr = np.zeros(n*k).reshape(n,k)

        # initial dummy assignment for first iteration
        last_assign = -np.ones(n)
        means = self.initial_values

        for iter in range(max_iter):

            # compute distance between each observation and each mean
            for c in range(k):
                dist_arr[:,c] = np.linalg.norm(means[c,:] - self.data, axis=1)

            # assign to nearest mean
            cur_assign = dist_arr.argmin(axis=1)

            # update means based on new assignments
            for j in range(k):
                means[j,:] = self.data[cur_assign == j,:].mean(axis=0)

            # termination block: only enter if we have hit local minima
            if np.array_equal(cur_assign, last_assign):
                self.cluster_assignments = cur_assign
                self.cluster_centers = means
                return None

            # update last iterations assignment for next comparison
            last_assign = dist_arr.argmin(axis=1)

        # warn the user that it did not converge
        warnings.warn("Failed to Converge", RuntimeWarning)
        return None

    def report(self):
        """
        reports a summary of cluster assignments

        Requires that self.data and self.cluster_assignments are initialized
        Updates self.cluster_summary attribute to contain cluster summary information            Updates self.assignments_summary attribute to show cluster point pairings

        Prints a plot to the screen if data is 2 dimensional

        Output: cluster_summary (pd data frame, printed to screen)
                assignments_summary (pd data frame)
                plot (if 2D data)
        """

        # cluster assignments must be initialized
        if self.cluster_assignments is None:
            raise ValueError("Cluster assignments must be assigned before plotting")

        # each point must have a cluster assignment
        if self.data.shape[0] != self.cluster_assignments.shape[0]:
            raise ValueError("Cluster assignenments and data are different lengths!")

        counts = []
        for k in range(0, self.K):
            counts.append(sum(self.cluster_assignments == k))

        self.cluster_summary = pd.DataFrame({'cluster' : list(range(0,self.K)),
                                             'count' : counts})

        self.assignment_summary = pd.DataFrame(self.data)
        self.assignment_summary['cluster'] = self.cluster_assignments

        if self.data.shape[1] == 2:
            self.assignment_summary = self.assignment_summary.rename(index=str,columns={0: "x", 1: "y"})
            sns.lmplot('x', 'y', data=self.assignment_summary, hue='cluster', fit_reg=False)
            plt.title("cluster assignments")

        print(self.cluster_summary)
