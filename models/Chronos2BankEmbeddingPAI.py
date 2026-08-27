import numpy as np
import math
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from TSB_AD.base import BaseDetector
from TSB_AD.utils.dataset import ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu

from models.Chronos2BankEmbeddingCosine import Chronos2
from models.bank_scoring import get_bank_embedding, compute_anomaly_score, score_against_bank
from models.pai import PAIScorer


class Chronos2BankEmbeddingPAI(BaseDetector):
    # euclidean bank distance on unnormalized embeddings + PAI amplitude scores (magG, T2);
    # lambda_g = lambda_q = 0 recovers a pure euclidean bank (cosine-vs-euclidean diagnostic)
    def __init__(self,
                 win_size = 100,
                 feats = 1,
                 batch_size = 256,
                 num_cores = 500,
                 top_k = 3,
                 w_b = 1.0,
                 lambda_g = 0.667,
                 lambda_q = 0.2,
                 pai_window = 32,
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.batch_size = batch_size
        self.feats = feats
        self.num_cores = num_cores
        self.top_k = top_k
        self.pai = PAIScorer(w_b=w_b, lambda_g=lambda_g, lambda_q=lambda_q, window=pai_window)

        self.model = Chronos2(window_size=win_size, n_vars=feats, device=self.device)

        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def fit(self, data):
        raw_data = data

        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        train_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        # device=None: batches stay on cpu, the Chronos-2 pipeline handles the device transfer itself
        bank_embedding, train_embeddings = get_bank_embedding(self.model, train_loader, num_cores=self.num_cores,
                                                              metric='euclidean', device=None, return_embeddings=True)
        self.bank_embedding = bank_embedding.to(self.device)

        # PAI z-score statistics come from the normal train split (paper recipe):
        # bank scores of the train windows + raw-series amplitude components
        train_scores = score_against_bank(train_embeddings, bank_embedding, metric='euclidean', top_k=self.top_k).numpy()
        self.pai.fit(raw_data, train_base_scores=train_scores)

    def decision_function(self, data):
        raw_data = data
        data = self.scaler.transform(data)

        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        scores = compute_anomaly_score(self.model, test_loader, self.bank_embedding,
                                       metric='euclidean', top_k=self.top_k, device=None)

        if scores.shape[0] < len(data):
            scores = np.array([scores[0]]*math.ceil((self.win_size-1)/2) +
                        list(scores) + [scores[-1]]*((self.win_size-1)//2))

        scores = self.pai.combine(raw_data, scores)

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
