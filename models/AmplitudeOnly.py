import numpy as np

from TSB_AD.base import BaseDetector

from models.pai import PAIScorer

# Amplitude-only diagnostic baseline: the PAI amplitude channels alone, no
# representation and no memory bank —
#   S(t) = lambda_g * z(magG) + lambda_q * z(T2)
# (a PAIScorer with w_b = 0). z-score statistics come from the normal train
# split. Measures how much of the benchmark the raw-series amplitude cues carry
# by themselves; any bank detector must clearly beat this to claim its
# representation adds value.


class AmplitudeOnly(BaseDetector):
    def __init__(self,
                 feats = 1,
                 lambda_g = 0.667,
                 lambda_q = 0.2,
                 pai_window = 32,
                 ):
        super().__init__()

        self.__anomaly_score = None
        self.pai = PAIScorer(w_b=0.0, lambda_g=lambda_g, lambda_q=lambda_q, window=pai_window)

    def fit(self, data):
        self.pai.fit(data)  # raw-series median/MAD + train-split z-score statistics

    def decision_function(self, data):
        # w_b = 0: the base term is inert, pass a zero placeholder
        scores = self.pai.combine(data, np.zeros(len(data)))

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
