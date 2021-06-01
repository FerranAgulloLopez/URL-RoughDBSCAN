from data import generate_blobs_dataset, generate_moons_dataset
import numpy as np
from visualize import plot_multiple_clustering_2d
import random
from sklearn.cluster import DBSCAN


random.seed(0)
np.random.seed(0)

dataset_blobs = generate_blobs_dataset(samples=1000, centers=10)
dataset_moons = generate_moons_dataset(samples=1000)

threshold_distance = [0.5, 1, 2, 4]
min_points = [2, 4, 8, 16]

outputs_blobs = []
outputs_moons = []
for index_1 in range(len(threshold_distance)):
    for index_2 in range(len(min_points)):
        algorithm = DBSCAN(eps=threshold_distance[index_1], min_samples=min_points[index_2])
        outputs_blobs.append(algorithm.fit_predict(dataset_blobs))
        outputs_moons.append(algorithm.fit_predict(dataset_moons))

plot_multiple_clustering_2d(len(threshold_distance), len(min_points), dataset_blobs, outputs_blobs, 'title', threshold_distance, min_points, './charts/second_part/blobs')
plot_multiple_clustering_2d(len(threshold_distance), len(min_points), dataset_moons, outputs_moons, 'title', threshold_distance, min_points, './charts/second_part/moons')

leaders_threshold_distance = [1, 2, 4]
