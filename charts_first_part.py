from algorithms.RoughDBSCAN import RoughDBSCAN
from data import generate_blobs_dataset, generate_moons_dataset
import numpy as np
from algorithms.RoughDBSCAN import RoughDBSCAN
from visualize import plot_multiple_clustering_2d
from time import time
import random


random.seed(0)
np.random.seed(0)

dataset_blobs = generate_blobs_dataset(samples=1000, centers=10)
dataset_moons = generate_moons_dataset(samples=1000)

threshold_distance = [0.5, 1, 2, 4]
min_points = [8, 16, 32, 64]

outputs_blobs = []
outputs_moons = []
for index_1 in range(len(threshold_distance)):
    for index_2 in range(len(min_points)):
        algorithm = RoughDBSCAN(threshold_distance_leaders=threshold_distance[index_1] - 0.45, threshold_distance=threshold_distance[index_1], min_points=min_points[index_2])
        outputs_blobs.append(algorithm.fit_predict(dataset_blobs))
        outputs_moons.append(algorithm.fit_predict(dataset_moons))

plot_multiple_clustering_2d(len(threshold_distance), len(min_points), dataset_blobs, outputs_blobs, 'title', threshold_distance, min_points, './charts/first_part/blobs')
plot_multiple_clustering_2d(len(threshold_distance), len(min_points), dataset_moons, outputs_moons, 'title', threshold_distance, min_points, './charts/first_part/moons')

leaders_threshold_distance = [1, 2, 4]

outputs_blobs = []
outputs_moons = []
for index in range(len(leaders_threshold_distance)):
    algorithm = RoughDBSCAN(threshold_distance_leaders=leaders_threshold_distance[index], threshold_distance=2, min_points=4)
    outputs_blobs.append(algorithm.fit_predict(dataset_blobs))
for index in range(len(leaders_threshold_distance)):
    algorithm = RoughDBSCAN(threshold_distance_leaders=leaders_threshold_distance[index], threshold_distance=2, min_points=4)
    outputs_moons.append(algorithm.fit_predict(dataset_moons))

plot_multiple_clustering_2d(1, len(leaders_threshold_distance), dataset_blobs, outputs_blobs, 'title', None, leaders_threshold_distance, './charts/first_part/leaders_blobs')
plot_multiple_clustering_2d(1, len(leaders_threshold_distance), dataset_moons, outputs_moons, 'title', None, leaders_threshold_distance, './charts/first_part/leaders_moons')
