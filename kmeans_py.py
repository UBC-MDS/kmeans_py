"""User-friendly k-means clustering package"""

def kmeans_init(data, K, algorithm = 'k-means++'):
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

Examples
--------
import numpy as np
data = np.random.uniform(0,10,100)
np.random.normal(5,1,100)
data = data.frame(x = runif(100, min = 0, max = 10) + rep(c(0, 10), 50), y = rnorm(100, 5, 1) + rep(c(0, 10), 50))

"""
# ` @examples
# ` # create input data object with two distinct clusters
# ` data = data.frame(x = runif(100, min = 0, max = 10) + rep(c(0, 10), 50), y = rnorm(100, 5, 1) + rep(c(0, 10), 50))
# `
# ` kmeans_init(data = data, K = 2)