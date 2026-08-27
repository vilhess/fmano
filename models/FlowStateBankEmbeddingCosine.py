import numpy as np
import math
import os
import sys
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from TSB_AD.base import BaseDetector
from TSB_AD.utils.dataset import ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu

from models.bank_scoring import get_bank_embedding, compute_anomaly_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'other_modules'))
from tsfm_public import FlowStateForPrediction
from einops import rearrange

class FlowState(nn.Module):
    def __init__(self, window_size, n_vars, device, scale_factor=1.0):
        super().__init__()
        # embeds the window; anomaly score = distance to the closest clean (bank) embeddings
        self.scale_factor = scale_factor
        self.model = FlowStateForPrediction.from_pretrained("ibm-research/flowstate", revision="r1.1").to(device)

    def forward(self, x):

        # Input:

        # x: bs x window_size x nvars
        bs, win_size, nvars = x.shape
        ctx = rearrange(x, "bs win_size nvars -> (bs nvars) win_size 1")
        out = self.model(ctx, scale_factor=self.scale_factor, prediction_length=1,
                         batch_first=True, prediction_type="median")
        emb = out.backbone_hidden_state[0]  # last S5 state before the decoder: bs*nvars x emb_dim
        emb = rearrange(emb, "(bs nvars) emb_dim -> bs (nvars emb_dim)", bs=bs, nvars=nvars)
        return emb.to(x.device)



class FlowStateBankEmbeddingCosine(BaseDetector):
    def __init__(self,
                 win_size = 100,
                 feats = 1,
                 batch_size = 256,
                 num_cores = 500,
                 top_k = 3,
                 metric = 'cosine',
                 scale_factor = 1.0,
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
        self.metric = metric

        self.model = FlowState(window_size=win_size, n_vars=feats, device=self.device,
                               scale_factor=scale_factor).to(self.device)

        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def fit(self, data):
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        train_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        bank_embedding = get_bank_embedding(self.model, train_loader, num_cores=self.num_cores, metric=self.metric, device=self.device)
        self.bank_embedding = bank_embedding.to(self.device)

    def decision_function(self, data):
        data = self.scaler.transform(data)

        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        scores = compute_anomaly_score(self.model, test_loader, self.bank_embedding,
                                       metric=self.metric, top_k=self.top_k, device=self.device)

        if scores.shape[0] < len(data):
            scores = np.array([scores[0]]*math.ceil((self.win_size-1)/2) +
                        list(scores) + [scores[-1]]*((self.win_size-1)//2))

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
