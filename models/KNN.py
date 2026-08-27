import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from TSB_AD.base import BaseDetector

# kNN anomaly detection directly in raw window space (no foundation model).
# Adapted from the PyOD/TSB-AD KNN baseline (Ramaswamy et al., 2000): the score of a
# window is a statistic of its distances to the k nearest train windows:
#   'largest': distance to the kth neighbor, 'mean'/'median': over the k neighbors.


def _sliding_windows(data, win_size):
    # data: n x feats -> (n - win_size + 1) x (win_size * feats)
    n, feats = data.shape
    windows = sliding_window_view(data, (win_size, feats))
    return windows.reshape(n - win_size + 1, win_size * feats)


class KNN(BaseDetector):
    def __init__(self,
                 win_size = 100,
                 feats = 1,
                 n_neighbors = 10,
                 method = 'largest',
                 n_jobs = -1,
                 ):
        super().__init__()

        assert method in ('largest', 'mean', 'median'), f"Unknown method: {method}"

        self.__anomaly_score = None

        self.win_size = win_size
        self.feats = feats
        self.n_neighbors = n_neighbors
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, data):
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        train_windows = _sliding_windows(data, self.win_size)
        k = min(self.n_neighbors, len(train_windows))
        if k < self.n_neighbors:
            print(f"n_neighbors ({self.n_neighbors}) is greater than the number of train windows ({len(train_windows)}), using k={k}.")
        self.neigh = NearestNeighbors(n_neighbors=k, n_jobs=self.n_jobs)
        self.neigh.fit(train_windows)

    def decision_function(self, data):
        data = self.scaler.transform(data)

        test_windows = _sliding_windows(data, self.win_size)
        dist_arr, _ = self.neigh.kneighbors(test_windows, return_distance=True)

        if self.method == 'largest':
            scores = dist_arr[:, -1]
        elif self.method == 'mean':
            scores = dist_arr.mean(axis=1)
        else:  # median
            scores = np.median(dist_arr, axis=1)

        if scores.shape[0] < len(data):
            scores = np.array([scores[0]]*math.ceil((self.win_size-1)/2) +
                        list(scores) + [scores[-1]]*((self.win_size-1)//2))

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
