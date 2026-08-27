import numpy as np

from TSB_AD.base import BaseDetector

EPS = 1e-9

def _flatten_signal(data):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[1] == 1:
        return x[:, 0]
    return np.mean(x, axis=1)

def _mean_std(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.nanmean(x)), float(np.nanstd(x) + EPS)

def centeredmwvar(query, window):
    # symmetric window of radius `window` around t (width 2W+1), edges use available samples
    n = len(query)
    radius = max(1, int(window))
    cumsum = np.concatenate([[0.0], np.cumsum(query, dtype=np.float64)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(query**2, dtype=np.float64)])
    t = np.arange(n)
    left = np.maximum(0, t - radius)
    right = np.minimum(n, t + radius + 1)
    win_mean = (cumsum[right] - cumsum[left]) / np.maximum(1, right - left)
    win_mean_sq = (cumsum_sq[right] - cumsum_sq[left]) / np.maximum(1, right - left)
    return win_mean_sq - win_mean**2

def centeredsq(query, window):
    # symmetric window of radius `window` around t (width 2W+1), edges use available samples
    n = len(query)
    radius = max(1, int(window))
    cumsum = np.concatenate([[0.0], np.cumsum(query, dtype=np.float64)])
    t = np.arange(n)
    left = np.maximum(0, t - radius)
    right = np.minimum(n, t + radius + 1)
    win_mean = (cumsum[right] - query - cumsum[left]) / np.maximum(1, right - left - 1)
    return (win_mean - query)**2

class OneLinerScorer:
    # window_mw / window_sq are independent: moving-window variance (Def. 4 in Zhu et al., 2026)
    # is evaluated with a large window matching TSFM context length in the paper, while the
    # centered squared-difference (their Centered-w) is evaluated with a small window (e.g.
    # Centered-5) — sharing one window between the two components conflates two different scales.
    def __init__(self, lambda_mw=0.667, lambda_sq=0.2, window_mw=32, window_sq=2):
        self.lambda_mw = lambda_mw
        self.lambda_sq = lambda_sq
        self.window_mw = window_mw
        self.window_sq = window_sq

    def fit(self, train_series):
        # train_series: RAW (unscaled) train split
        x = _flatten_signal(train_series)

        # z-score statistics of each component on the normal train split
        self.mwvar_mean, self.mwvar_std = _mean_std(centeredmwvar(x, self.window_mw))
        self.sq_mean, self.sq_std = _mean_std(centeredsq(x, self.window_sq))

        return self

    def combine(self, test_series):
        # test_series: RAW (unscaled) test series
        query = _flatten_signal(test_series)

        mwvars = centeredmwvar(query, self.window_mw)
        sq = centeredsq(query, self.window_sq)

        score = (
                 self.lambda_mw * (mwvars - self.mwvar_mean) / self.mwvar_std
                 + self.lambda_sq * (sq - self.sq_mean) / self.sq_std)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


class OneLiner(BaseDetector):
    def __init__(self,
                 feats = 1,
                 lambda_mw = 0.667,
                 lambda_sq = 0.2,
                 window_mw = 32,
                 window_sq = 2,
                 ):
        super().__init__()

        self.__anomaly_score = None
        self.oneliner = OneLinerScorer(lambda_mw=lambda_mw, lambda_sq=lambda_sq,
                                        window_mw=window_mw, window_sq=window_sq)

    def fit(self, data):
        self.oneliner.fit(data)

    def decision_function(self, data):
        scores = self.oneliner.combine(data)

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
