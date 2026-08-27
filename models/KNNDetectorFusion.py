import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from TSB_AD.base import BaseDetector

from models.KNN import _sliding_windows
from models.pai import _mean_std

# Score-level fusion of KNN (raw window space) with any full black-box detector:
#   S(t) = z(KNN) + lambda_bank * z(detector)
# Used when the fusion partner's own score already combines multiple signals (e.g.
# a memory bank fused with PAI amplitude channels), so there is no single bare
# embedding/bank call to hook into (unlike KNNBankFusion) — the wrapped detector is
# only used through its public fit()/decision_function(). Both terms use statistics
# of the normal train split: self-excluded kNN distances of the train windows for
# the KNN term, the wrapped detector's own train-split scores for the other term.
# lambda_bank = 0 recovers pure KNN.
#
# Subclasses implement _build_detector() (returning the unfitted wrapped BaseDetector
# instance; self.feats is already set when it's called).


class KNNDetectorFusion(BaseDetector):
    def __init__(self,
                 win_size = 64,
                 feats = 1,
                 n_neighbors = 50,
                 method = 'mean',
                 lambda_bank = 1.0,
                 n_jobs = -1,
                 ):
        super().__init__()

        assert method in ('largest', 'mean', 'median'), f"Unknown method: {method}"

        self.__anomaly_score = None

        self.win_size = win_size
        self.feats = feats
        self.n_neighbors = n_neighbors
        self.method = method
        self.lambda_bank = lambda_bank
        self.n_jobs = n_jobs

        self.detector = self._build_detector()

    def _build_detector(self):
        raise NotImplementedError

    def _dist_by_method(self, dist_arr):
        if self.method == 'largest':
            return dist_arr[:, -1]
        elif self.method == 'mean':
            return dist_arr.mean(axis=1)
        return np.median(dist_arr, axis=1)

    def fit(self, data):
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(data)

        # KNN component: raw windows; z-stats from self-excluded train kNN distances
        train_windows = _sliding_windows(scaled, self.win_size)
        k = min(self.n_neighbors, max(1, len(train_windows) - 1))
        if k < self.n_neighbors:
            print(f"n_neighbors ({self.n_neighbors}) is greater than the number of train windows - 1 ({len(train_windows) - 1}), using k={k}.")
        self.neigh = NearestNeighbors(n_neighbors=k, n_jobs=self.n_jobs)
        self.neigh.fit(train_windows)
        train_dists, _ = self.neigh.kneighbors(return_distance=True)
        self.knn_mean, self.knn_std = _mean_std(self._dist_by_method(train_dists))

        # wrapped detector component: full black-box detector on the raw (unscaled)
        # split; z-stats from its own train-split scores
        self.detector.fit(data)
        train_detector_scores = self.detector.decision_function(data)
        self.detector_mean, self.detector_std = _mean_std(train_detector_scores)

    def decision_function(self, data):
        scaled = self.scaler.transform(data)
        n_points = len(data)

        test_windows = _sliding_windows(scaled, self.win_size)
        dist_arr, _ = self.neigh.kneighbors(test_windows, return_distance=True)
        knn_scores = self._dist_by_method(dist_arr)
        if knn_scores.shape[0] < n_points:
            knn_scores = np.array([knn_scores[0]]*math.ceil((self.win_size-1)/2) +
                        list(knn_scores) + [knn_scores[-1]]*((self.win_size-1)//2))

        detector_scores = self.detector.decision_function(data)

        scores = ((knn_scores - self.knn_mean) / self.knn_std
                  + self.lambda_bank * (detector_scores - self.detector_mean) / self.detector_std)

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
