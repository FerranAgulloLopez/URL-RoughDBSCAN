import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from scipy.spatial import distance
from src.algorithms.types.CountedLeaders import CountedLeaders
from collections import deque


class RoughDBSCAN(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, threshold_distance_leaders, threshold_distance, min_points):
        self.threshold_distance = threshold_distance
        self.min_points = min_points
        self.counted_leaders_algorithm = CountedLeaders(threshold_distance_leaders)

    # Shared clustering methods

    def fit(self, X):
        self.fit_predict(X)  # makes not much sense otherwise
        return self

    def fit_predict(self, X, y=None):
        samples_leaders = self.counted_leaders_algorithm.fit_transform(X)
        leaders_values = np.asarray(self.counted_leaders_algorithm.get_leaders_values())
        leaders_followers_indexes = np.asarray(self.counted_leaders_algorithm.get_leaders_followers_indexes())

        cluster_id = 0
        leaders_clusters = np.zeros(leaders_values.shape[0])
        leaders_seen = np.zeros(leaders_values.shape[0])
        distances_between_leaders = distance.cdist(leaders_values, leaders_values, 'euclidean')

        for leader_index in range(leaders_values.shape[0]):
            if leaders_seen[leader_index] == 0:
                leaders_seen[leader_index] = 1
                cardinality, close_leaders_indexes = self._compute_leader_cardinality(leader_index, distances_between_leaders, leaders_followers_indexes)
                if cardinality < self.min_points:
                    leaders_clusters[leader_index] = -1
                else:
                    leaders_clusters[close_leaders_indexes] = cluster_id
                    not_seen_leader_indexes = deque(np.where(leaders_seen[close_leaders_indexes] == 0)[0])
                    while not_seen_leader_indexes:
                        not_seen_leader_index = not_seen_leader_indexes.pop()
                        leaders_seen[not_seen_leader_index] = 1
                        cardinality, close_leaders_indexes = self._compute_leader_cardinality(leader_index, distances_between_leaders, leaders_followers_indexes)
                        if cardinality > self.min_points:
                            leaders_clusters[close_leaders_indexes] = cluster_id
                            not_seen_leader_indexes.extend(np.where(leaders_seen[close_leaders_indexes] == 0)[0])
                    cluster_id += 1

        output = np.zeros(X.shape[0])
        for index in range(output.shape[0]):
            output[index] = leaders_clusters[int(samples_leaders[index])]

        return self

    # Private methods

    def _compute_leader_cardinality(self, leader_index, distances_between_leaders, leaders_followers_indexes):
        close_leaders_indexes = self._get_close_leaders(leader_index, distances_between_leaders)
        return sum([len(leaders_followers_indexes[index]) for index in close_leaders_indexes]), close_leaders_indexes

    def _get_close_leaders(self, leader_index, distances_between_leaders):
        return np.where(distances_between_leaders[leader_index] <= self.threshold_distance)[0]
