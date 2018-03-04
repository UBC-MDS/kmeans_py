"""User-friendly k-means clustering package"""
import numpy as np

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
        if self.data is None or type(self.data) is not np.ndarray:
            raise TypeError("Data is missing or of wrong object type. Please specify data as a Numpy array.")

        if self.K is None or self.K%1 != 0:
            raise TypeError("K is missing or is not an integer. Please specify K as an integer.")

        if self.data.shape[0] < self.K:
            raise ValueError("Cannot choose more initialize values than data observations.")

        pass

    def cluster_points(self):
        """
        Perform k-means clustering on the provided data

        Requires that kmeans.intialize_centers() has been run in advance

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
        Plot the clustered points, colour by cluster assignments

        Requires that self.data and self.cluster_assignments are initialized
        Prints a plot to the screen & saves plot as an image in the root directory.

        Output: image file "kmeans_plot.png" in root directory
        """
        if self.cluster_assignments == None:
            raise ValueError("Cluster assignments must be assigned before plotting")


        pass
