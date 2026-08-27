import numpy as np
import math
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import tqdm

from TSB_AD.base import BaseDetector
from TSB_AD.utils.dataset import ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'other_modules'))
from tsfm_public import FlowStateForPrediction
from einops import rearrange

class FlowState(nn.Module):
    def __init__(self, window_size, n_vars, device, scale_factor=1.0):
        super().__init__()
        # forecasts the last time step from the window_size-1 previous ones
        self.scale_factor = scale_factor
        self.model = FlowStateForPrediction.from_pretrained("ibm-research/flowstate", revision="r1.1").to(device)

    def forward(self, x):

        # Input:

        # x: bs x (window_size-1) x nvars
        bs, win_size, nvars = x.shape
        ctx = rearrange(x, "bs win_size nvars -> (bs nvars) win_size 1")
        out = self.model(ctx, scale_factor=self.scale_factor, prediction_length=1,
                         batch_first=True, prediction_type="median")
        preds = out.prediction_outputs  # bs*nvars x 1 x 1
        preds = preds.reshape(bs, nvars)
        return preds.to(x.device)  # bs x nvars

    def get_ano_score(self, x):

        # Input:

        # x: bs x window_size x nvars
        ctx = x[:, :-1, :]
        target = x[:, -1, :]
        pred = self.forward(ctx)
        return F.mse_loss(pred, target, reduction='none').mean(dim=1)  # bs


class FlowStateForecaster(BaseDetector):
    def __init__(self,
                 win_size = 100,
                 feats = 1,
                 batch_size = 256,
                 scale_factor = 1.0,
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.batch_size = batch_size
        self.feats = feats

        self.model = FlowState(window_size=win_size, n_vars=feats, device=self.device,
                               scale_factor=scale_factor).to(self.device)

        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def fit(self, data):
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def decision_function(self, data):
        data = self.scaler.transform(data)

        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()
        scores = []
        loop = tqdm.tqdm(test_loader, total=len(test_loader), leave=True)
        with torch.inference_mode():
            for x, _ in loop:
                x = x.to(self.device)  # bs x win_size x feats
                scores.append(self.model.get_ano_score(x).cpu())

        scores = torch.cat(scores, dim=0).numpy()

        if scores.shape[0] < len(data):
            scores = np.array([scores[0]]*(self.win_size-1) + list(scores))

        assert len(scores) == len(data), f"Scores length {len(scores)} does not match data length {len(data)}"

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
