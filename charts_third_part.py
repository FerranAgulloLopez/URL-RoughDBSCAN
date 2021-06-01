from algorithms.RoughDBSCAN import RoughDBSCAN
from algorithms.CountedLeadersStep import CountedLeadersStep
from algorithms.CountedLeadersFull import CountedLeadersFull
from algorithms.CountedLeadersStructureStep import CountedLeadersStructureStep
from algorithms.CountedLeadersStructureFull import CountedLeadersStructureFull
from data import generate_blobs_dataset
import numpy as np
from algorithms.RoughDBSCAN import RoughDBSCAN
from visualize import plot_multiple_clustering_2d
from time import time
import random


random.seed(0)
np.random.seed(0)

dataset_blobs = generate_blobs_dataset(samples=1000, centers=10)

leaders_algorithm = [CountedLeadersStep, CountedLeadersFull, CountedLeadersStructureStep, CountedLeadersStructureFull]

outputs = []
for index in range(len(leaders_algorithm)):
    algorithm = RoughDBSCAN(threshold_distance_leaders=2, threshold_distance=3, min_points=4, counted_leaders=leaders_algorithm[index])
    outputs.append(algorithm.fit_predict(dataset_blobs))

plot_multiple_clustering_2d(1, len(leaders_algorithm), dataset_blobs, outputs, 'title', None, ['CountedLeadersStep', 'CountedLeadersFull', 'CountedLeadersStructureStep', 'CountedLeadersStructureFull'], './charts/third_part/leaders')
