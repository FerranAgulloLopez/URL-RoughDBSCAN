import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from scipy.spatial import distance
from collections import deque


class OriginalDBSCAN(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, threshold_distance, min_points):
        self.threshold_distance = threshold_distance
        self.min_points = min_points

    # Shared clustering methods

    def fit(self, X):
        self.fit_predict(X)  # makes not much sense otherwise (following the DBSCAN fashion)
        return self

    def fit_predict(self, X, y=None):
        cluster_id = 0
        samples_clusters = np.zeros(X.shape[0])
        samples_seen = np.zeros(X.shape[0])
        distances_between_samples = distance.squareform(distance.pdist(X, 'euclidean'))

        for sample_index in range(X.shape[0]):
            if samples_seen[sample_index] == 0:
                samples_seen[sample_index] = 1
                close_samples_indexes = self._get_close_samples(sample_index, distances_between_samples)
                if len(close_samples_indexes) < self.min_points:
                    samples_clusters[sample_index] = -1
                else:
                    samples_clusters[close_samples_indexes] = cluster_id
                    not_seen_samples_indexes = deque(close_samples_indexes[np.where(samples_seen[close_samples_indexes] == 0)])
                    while not_seen_samples_indexes:
                        not_seen_sample_index = not_seen_samples_indexes.pop()
                        samples_seen[not_seen_sample_index] = 1
                        close_samples_indexes = self._get_close_samples(sample_index, distances_between_samples)
                        if len(close_samples_indexes) < self.min_points:
                            samples_clusters[close_samples_indexes] = cluster_id
                            not_seen_samples_indexes.extend(close_samples_indexes[np.where(samples_seen[close_samples_indexes] == 0)])
                    cluster_id += 1

        return samples_clusters

    # Private methods

    def _get_close_samples(self, leader_index, distances_between_leaders):
        return np.where(distances_between_leaders[leader_index] <= self.threshold_distance)[0]
