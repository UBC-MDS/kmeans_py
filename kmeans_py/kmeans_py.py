"""User-friendly k-means clustering package"""


class kmeans():

    def __init__(self, data, K):

        self.data = data
        self.K = K
        self.initial_values = None
        self.cluster_assignments = None


    def initialize_centers(self, algorithm = 'k-means++'):
        """ Choose Initial K-Means Values

        Arguments
        ---------
        data: array
            the data object that k-means clustering will be applied to.

        K: float
            the number of initial values to be chosen. Should correspond to the number of clusters to be chosen.

        algorithm: string (default = 'k-means++')
            the initialisation algorithm specified as a string.

            - 'k-means++': K-means++ optimization algorithm. Safer, but more time complex, initialization algorithm compared to Lloyd's algorithm.

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
        pass

    def cluster_points(self):
        """
        Perform k-means clustering on the provided data

        Inputs:
         - data:    an n x d array of data points to be clustered
         - centers: a k x d array of centers (means) for intialization

        Output:
         - an n x 1 array of hard cluster assignments
        """
        if self.data.shape[1] != self.initial_values.shape[1]:
            raise ValueError("Initial values and data are not compatible shape")

        pass

    def plot(self):
        """
        Plot the clustered points
        """

        pass
