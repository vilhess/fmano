import numpy as np

# PAI: Preserving Amplitude Information (arXiv:2606.08935), score-level extension for
# representation-based detectors. Follows the official PaAno_PAI code (A28f recipe):
#   S_PAI(t) = w_b * z(base) + lambda_g * z(magG) + lambda_q * z(T2)
# magG(t) = |x_t - median(train)| / MAD(train) catches spikes,
# T2(t) = |mean(x_{t-W..t+W}) - median(train)| / MAD(train) catches level shifts.
# All components operate on the RAW series (median/MAD make them scale-free) and each
# component is z-scored with the statistics of the NORMAL TRAIN split (paper recipe):
# magG/T2 stats come from the train series, base stats from the bank scores of the
# train windows. Falls back to self-normalization if train base scores are not given.

EPS = 1e-9


def _flatten_signal(data):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[1] == 1:
        return x[:, 0]
    return np.mean(x, axis=1)


def _zscore(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    scale = float(np.nanstd(x) + EPS)
    return (x - float(np.nanmean(x))) / scale


def _mean_std(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.nanmean(x)), float(np.nanstd(x) + EPS)


def _mag_g(query, median, mad):
    return np.abs(query - median) / mad


def _t2(query, median, mad, window):
    # symmetric window of radius `window` around t (width 2W+1), edges use available samples
    n = len(query)
    radius = max(1, int(window))
    cumsum = np.concatenate([[0.0], np.cumsum(query, dtype=np.float64)])
    t = np.arange(n)
    left = np.maximum(0, t - radius)
    right = np.minimum(n, t + radius + 1)
    win_mean = (cumsum[right] - cumsum[left]) / np.maximum(1, right - left)
    return np.abs(win_mean - median) / mad


class PAIScorer:
    def __init__(self, w_b=1.0, lambda_g=0.667, lambda_q=0.2, window=32):
        self.w_b = w_b
        self.lambda_g = lambda_g
        self.lambda_q = lambda_q
        self.window = window

    def fit(self, train_series, train_base_scores=None):
        # train_series: RAW (unscaled) train split
        # train_base_scores: bank scores of the train windows, for the base z-score statistics
        x = _flatten_signal(train_series)
        self.median = float(np.median(x))
        self.mad = float(np.median(np.abs(x - self.median)) + EPS)

        # z-score statistics of each component on the normal train split
        self.magg_mean, self.magg_std = _mean_std(_mag_g(x, self.median, self.mad))
        self.t2_mean, self.t2_std = _mean_std(_t2(x, self.median, self.mad, self.window))
        if train_base_scores is not None:
            self.base_mean, self.base_std = _mean_std(train_base_scores)
        else:
            self.base_mean = self.base_std = None
        return self

    def combine(self, test_series, base_scores):
        # test_series: RAW (unscaled) test series
        # base_scores: per-point bank scores, already padded to len(test_series)
        query = _flatten_signal(test_series)
        base = np.asarray(base_scores, dtype=np.float64).reshape(-1)
        assert len(base) == len(query), \
            f"base_scores ({len(base)}) and series ({len(query)}) must have the same length"

        mag_g = _mag_g(query, self.median, self.mad)
        t2 = _t2(query, self.median, self.mad, self.window)

        if self.base_mean is None:
            base_z = _zscore(base)  # fallback: self-normalization on the scored series
        else:
            base_z = (base - self.base_mean) / self.base_std

        score = (self.w_b * base_z
                 + self.lambda_g * (mag_g - self.magg_mean) / self.magg_std
                 + self.lambda_q * (t2 - self.t2_mean) / self.t2_std)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
