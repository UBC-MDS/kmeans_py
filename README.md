<img src="docs/images/logo_py_crop.png" align="right" border = "10" />

# kmeans_py

[![GitHub issues](https://img.shields.io/github/issues/UBC-MDS/kmeans_py.svg)](https://github.com/UBC-MDS/kmeans_py/issues)

## Overview

**kmeans_py** is an Python package aimed towards a user-friendly way of exploring and implementing k-means clustering.

The package integrates and simplifies different functions, such as [sklearn's KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [scipy's kmeans](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.cluster.vq.kmeans.html), into one easy-to-use package.

The package includes the following functions:

* `kmeans++(data, K, algorithm = "kmeans++")` Creating initialization values. By default implements [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization algorithm. Returns `K` number of initial centers based on the input `data`.

* `kmeans(data, centers_init)` Classifies each observation in `data` by performing [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering). The number of clusters is derived from the number of initial centers specified in `centers_init`. Returns an object containing the original data and assigned cluster labels.

* `kmeans_plot(obj)` Visualizes clustered data using an object that is formatted in the same way as the object returned by the `kmeans` function.

## Contributors

[Bradley Pick](https://github.com/bradleypick)

[Charley Carriero](https://github.com/charcarr)

[Johannes Harmse](https://github.com/johannesharmse)
