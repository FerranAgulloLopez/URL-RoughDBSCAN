from algorithms.CountedLeadersFull import CountedLeadersFull
from algorithms.RoughDBSCAN import RoughDBSCAN
from data import generate_banana2, generate_gaussian_mixture
import numpy as np
from sklearn.cluster import DBSCAN
from algorithms.OriginalDBSCAN import OriginalDBSCAN
from visualize import plot_clustering_2d
from time import time

"""""
X = np.asarray([[0], [4], [1], [4], [50], [5]])
algorithm = RoughDBSCAN(3, 4, 3)
algorithm.fit(X)

"""""
X_banana2 = generate_gaussian_mixture()

init = time()
algorithm = DBSCAN(eps=0.43, min_samples=7)
y = algorithm.fit_predict(X_banana2)
print('Normal time:', time() - init)

init = time()
algorithm = RoughDBSCAN(threshold_distance_leaders=3, threshold_distance=9, min_points=3)
y = algorithm.fit_predict(X_banana2)
print('Rough time:', time() - init)

init = time()
algorithm = OriginalDBSCAN(9, 10)
y = algorithm.fit_predict(X_banana2)
print('Original time:', time() - init)

print(X_banana2.shape, y.shape)

plot_clustering_2d(X_banana2, y)
