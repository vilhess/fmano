import numpy as np
import math
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from TSB_AD.base import BaseDetector
from TSB_AD.utils.dataset import ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu

from models.KNN import _sliding_windows
from models.bank_scoring import get_bank_embedding, compute_anomaly_score, score_against_bank
from models.pai import _mean_std

# Score-level fusion of KNN (raw window space) with a TSFM cosine memory bank:
#   S(t) = z(KNN) + lambda_bank * z(BankEmbeddingCosine)
# KNN has the cleaner top of the ranking (VUS-PR), the TSFM cosine bank the better
# global separation (VUS-ROC) — complementary error profiles. Both z-normalizations
# use statistics of the normal train split: self-excluded kNN distances of the train
# windows for the KNN term, bank scores of the train windows for the bank term.
# lambda_bank = 0 recovers pure KNN.
#
# Subclasses only need to implement _build_model() (returning the *BankEmbeddingCosine
# encoder for their backbone) and, if the encoder's pipeline expects cpu batches and
# handles the device transfer itself (e.g. Chronos2), set _encode_on_cpu = True.


class KNNBankFusion(BaseDetector):
    _encode_on_cpu = False

    def __init__(self,
                 win_size = 64,
                 feats = 1,
                 batch_size = 256,
                 n_neighbors = 50,
                 method = 'mean',
                 num_cores = 500,
                 top_k = 3,
                 lambda_bank = 1.0,
                 n_jobs = -1,
                 ):
        super().__init__()

        assert method in ('largest', 'mean', 'median'), f"Unknown method: {method}"

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.feats = feats
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.method = method
        self.num_cores = num_cores
        self.top_k = top_k
        self.lambda_bank = lambda_bank
        self.n_jobs = n_jobs

        self.model = self._build_model()

        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def _build_model(self):
        raise NotImplementedError

    def _bank_device(self):
        return None if self._encode_on_cpu else self.device

    def _dist_by_method(self, dist_arr):
        if self.method == 'largest':
            return dist_arr[:, -1]
        elif self.method == 'mean':
            return dist_arr.mean(axis=1)
        return np.median(dist_arr, axis=1)

    def _pad(self, scores, n_points):
        if scores.shape[0] < n_points:
            scores = np.array([scores[0]]*math.ceil((self.win_size-1)/2) +
                        list(scores) + [scores[-1]]*((self.win_size-1)//2))
        return scores

    def fit(self, data):
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        # KNN component: raw windows; z-stats from self-excluded train kNN distances
        train_windows = _sliding_windows(data, self.win_size)
        k = min(self.n_neighbors, max(1, len(train_windows) - 1))
        if k < self.n_neighbors:
            print(f"n_neighbors ({self.n_neighbors}) is greater than the number of train windows - 1 ({len(train_windows) - 1}), using k={k}.")
        self.neigh = NearestNeighbors(n_neighbors=k, n_jobs=self.n_jobs)
        self.neigh.fit(train_windows)
        train_dists, _ = self.neigh.kneighbors(return_distance=True)
        self.knn_mean, self.knn_std = _mean_std(self._dist_by_method(train_dists))

        # bank component: TSFM cosine bank; z-stats from the train windows' bank scores
        train_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )
        bank_embedding, train_embeddings = get_bank_embedding(self.model, train_loader, num_cores=self.num_cores,
                                                              metric='cosine', device=self._bank_device(), return_embeddings=True)
        self.bank_embedding = bank_embedding.to(self.device)
        train_bank_scores = score_against_bank(train_embeddings, bank_embedding, metric='cosine', top_k=self.top_k).numpy()
        self.bank_mean, self.bank_std = _mean_std(train_bank_scores)

    def decision_function(self, data):
        data = self.scaler.transform(data)
        n_points = len(data)

        test_windows = _sliding_windows(data, self.win_size)
        dist_arr, _ = self.neigh.kneighbors(test_windows, return_distance=True)
        knn_scores = self._pad(self._dist_by_method(dist_arr), n_points)

        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )
        bank_scores = self._pad(compute_anomaly_score(self.model, test_loader, self.bank_embedding,
                                                      metric='cosine', top_k=self.top_k, device=self._bank_device()), n_points)

        scores = ((knn_scores - self.knn_mean) / self.knn_std
                  + self.lambda_bank * (bank_scores - self.bank_mean) / self.bank_std)

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
