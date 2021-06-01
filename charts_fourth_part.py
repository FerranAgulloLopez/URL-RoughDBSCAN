from algorithms.CountedLeadersStep import CountedLeadersStep
from algorithms.CountedLeadersFull import CountedLeadersFull
from algorithms.CountedLeadersStructureStep import CountedLeadersStructureStep
from algorithms.CountedLeadersStructureFull import CountedLeadersStructureFull
from data import generate_blobs_dataset, generate_moons_dataset
import numpy as np
from algorithms.RoughDBSCAN import RoughDBSCAN
from algorithms.OriginalDBSCAN import OriginalDBSCAN
from sklearn.cluster import DBSCAN
from time import time
from visualize import compare_multiple_lines
import random


random.seed(0)
np.random.seed(0)

dataset_samples = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
datasets_blobs = []
datasets_moons = []
for index in range(len(dataset_samples)):
    datasets_blobs.append(generate_blobs_dataset(samples=dataset_samples[index], centers=dataset_samples[index]//100))
    datasets_moons.append(generate_moons_dataset(samples=dataset_samples[index]))

all_lines_blobs = []
all_lines_moons = []

print('Rough-DBSCAN, CountedLeadersStep')
lines_blobs = []
lines_moons = []
for index in range(len(dataset_samples)):
    algorithm = RoughDBSCAN(threshold_distance_leaders=2, threshold_distance=3, min_points=4, counted_leaders=CountedLeadersStep)
    init_time = time()
    algorithm.fit_predict(datasets_blobs[index])
    lines_blobs.append(time() - init_time)
    init_time = time()
    algorithm.fit_predict(datasets_moons[index])
    lines_moons.append(time() - init_time)
all_lines_blobs.append(('CountedLeadersStep', lines_blobs))
all_lines_moons.append(('CountedLeadersStep', lines_moons))

print('Rough-DBSCAN, CountedLeadersFull')
lines_blobs = []
lines_moons = []
for index in range(len(dataset_samples)):
    if index < 5:
        algorithm = RoughDBSCAN(threshold_distance_leaders=2, threshold_distance=3, min_points=4, counted_leaders=CountedLeadersFull)
        init_time = time()
        algorithm.fit_predict(datasets_blobs[index])
        lines_blobs.append(time() - init_time)
        init_time = time()
        algorithm.fit_predict(datasets_moons[index])
        lines_moons.append(time() - init_time)
all_lines_blobs.append(('CountedLeadersFull', lines_blobs))
all_lines_moons.append(('CountedLeadersFull', lines_moons))

print('Rough-DBSCAN, CountedLeadersStructureStep')
lines_blobs = []
lines_moons = []
for index in range(len(dataset_samples)):
    algorithm = RoughDBSCAN(threshold_distance_leaders=2, threshold_distance=3, min_points=4, counted_leaders=CountedLeadersStructureStep)
    init_time = time()
    algorithm.fit_predict(datasets_blobs[index])
    lines_blobs.append(time() - init_time)
    init_time = time()
    algorithm.fit_predict(datasets_moons[index])
    lines_moons.append(time() - init_time)
all_lines_blobs.append(('CountedLeadersStructureStep', lines_blobs))
all_lines_moons.append(('CountedLeadersStructureStep', lines_moons))

print('Rough-DBSCAN, CountedLeadersStructureFull')
lines_blobs = []
lines_moons = []
for index in range(len(dataset_samples)):
    if index < 5:
        algorithm = RoughDBSCAN(threshold_distance_leaders=2, threshold_distance=3, min_points=4, counted_leaders=CountedLeadersStructureFull)
        init_time = time()
        algorithm.fit_predict(datasets_blobs[index])
        lines_blobs.append(time() - init_time)
        init_time = time()
        algorithm.fit_predict(datasets_moons[index])
        lines_moons.append(time() - init_time)
all_lines_blobs.append(('CountedLeadersStructureFull', lines_blobs))
all_lines_moons.append(('CountedLeadersStructureFull', lines_moons))

print('Sklearn DBSCAN')
lines_blobs = []
lines_moons = []
for index in range(len(dataset_samples)):
    if index < 5:
        algorithm = DBSCAN(eps=3, min_samples=4)
        init_time = time()
        algorithm.fit_predict(datasets_blobs[index])
        lines_blobs.append(time() - init_time)
        init_time = time()
        algorithm.fit_predict(datasets_moons[index])
        lines_moons.append(time() - init_time)
all_lines_blobs.append(('SklearnDBSCAN', lines_blobs))
all_lines_moons.append(('SklearnDBSCAN', lines_moons))

print('Original DBSCAN')
lines_blobs = []
lines_moons = []
for index in range(len(dataset_samples)):
    if index < 5:
        algorithm = OriginalDBSCAN(threshold_distance=3, min_points=4)
        init_time = time()
        algorithm.fit_predict(datasets_blobs[index])
        lines_blobs.append(time() - init_time)
        init_time = time()
        algorithm.fit_predict(datasets_moons[index])
        lines_moons.append(time() - init_time)
all_lines_blobs.append(('OriginalDBSCAN', lines_blobs))
all_lines_moons.append(('OriginalDBSCAN', lines_moons))


compare_multiple_lines(True, dataset_samples, all_lines_blobs, './charts/fourth_part/blobs', ylabel='Time (s)')
compare_multiple_lines(True, dataset_samples, all_lines_moons, './charts/fourth_part/moons', ylabel='Time (s)')
