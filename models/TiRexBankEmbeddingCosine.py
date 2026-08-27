import numpy as np
import math
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from TSB_AD.base import BaseDetector
from TSB_AD.utils.dataset import ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu

from models.bank_scoring import get_bank_embedding, compute_anomaly_score

from tirex import load_model, ForecastModel
from einops import rearrange

class TiRex(nn.Module):
    def __init__(self, window_size, n_vars, device):
        super().__init__()
        # embeds the window; anomaly score = distance to the closest clean (bank) embeddings
        self.model: ForecastModel = load_model("NX-AI/TiRex", device=str(device))

    def forward(self, x):

        # Input:

        # x: bs x window_size x nvars
        bs, win_size, nvars = x.shape
        ctx = rearrange(x, "bs win_size nvars -> (bs nvars) win_size")
        emb = self.model._embed_context(ctx)  # bs*nvars x n_tokens x n_layers x emb_dim
        emb = emb[:, -1, -1, :]  # last token, last layer: bs*nvars x emb_dim
        emb = rearrange(emb, "(bs nvars) emb_dim -> bs (nvars emb_dim)", bs=bs, nvars=nvars)
        return emb.to(x.device)


class TiRexBankEmbeddingCosine(BaseDetector):
    def __init__(self,
                 win_size = 100,
                 feats = 1,
                 batch_size = 256,
                 num_cores = 500,
                 top_k = 3,
                 metric = 'cosine',
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

        self.model = TiRex(window_size=win_size, n_vars=feats, device=self.device).to(self.device)

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
