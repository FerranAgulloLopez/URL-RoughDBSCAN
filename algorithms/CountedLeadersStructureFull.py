import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.neighbors import BallTree
from time import time


class CountedLeadersStructureFull(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, threshold_distance):
        self.threshold_distance = threshold_distance
        self.leaders_indexes = None
        self.leaders_values = None
        self.leaders_followers_indexes = None
        self.samples_leaders = None

    # Shared clustering methods

    def fit(self, X):
        self.fit_predict(X)  # the computation of the predict is minimal so we can reuse it
        return self

    def fit_predict(self, X, y=None):
        self.leaders_indexes = set()
        self.leaders_values = []  # array to store the leaders values
        self.leaders_followers_indexes = []  # array to store the leaders followers indexes
        self.samples_leaders = np.zeros(X.shape[0])  # array to store the leader index that each sample pertain to
        ball_tree = BallTree(X, metric='euclidean')

        init_time = time()
        if X.shape[0] > 0:
            # initialize arrays with the first sample
            self.leaders_indexes.add(0)
            self.leaders_values.append(X[0])
            self.leaders_followers_indexes.append([0])
            self.samples_leaders[0] = 0
            cluster_id = 1

            # iterate through all data samples

            for index in range(1, X.shape[0]):
                # find the leader of the sample
                sample_leader_index = self._get_sample_leader_index(X[index], ball_tree)
                if sample_leader_index == -1:
                    # the sample has no leader
                    # create a new leader with it
                    self.leaders_indexes.add(index)
                    self.leaders_values.append(X[index])
                    self.leaders_followers_indexes.append([index])
                    self.samples_leaders[index] = cluster_id
                    cluster_id += 1
                else:
                    # the sample has a leader
                    # append sample to its leader
                    self.leaders_followers_indexes[sample_leader_index].append(index)
                    self.samples_leaders[index] = sample_leader_index


        return self.samples_leaders

    def predict(self, X):
        # return the leader for each data sample, -1 if it does not exist
        samples_leaders = np.zeros(X.shape[0])
        for index in range(X.shape[0]):
            samples_leaders[index] = self._get_sample_leader_index(X[index])  # TODO update this
        return samples_leaders

    # TODO do partial fit

    # Unique methods

    def get_leaders_values(self):
        return self.leaders_values

    def get_leaders_followers_indexes(self):
        return self.leaders_followers_indexes

    # Private methods

    def _get_sample_leader_index(self, sample, ball_tree):
        init_time = time()
        aux = ball_tree.query_radius([sample], self.threshold_distance)[0]
        print('meh1:', time() - init_time)
        init_time = time()
        close_leaders_indexes = self.leaders_indexes.intersection(set(aux))
        print('meh2:', time() - init_time)
        if len(close_leaders_indexes) > 0:
            return int(self.samples_leaders[close_leaders_indexes.pop()])
        else:
            return -1
