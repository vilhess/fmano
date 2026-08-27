import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from TSB_AD.base import BaseDetector
from TSB_AD.utils.dataset import ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu

from models.bank_scoring import get_bank_embedding, compute_anomaly_score

from chronos import BaseChronosPipeline, Chronos2Pipeline
from einops import rearrange

class Chronos2(nn.Module):
    def __init__(self, window_size, n_vars, device):
        super().__init__()
        # embeds the window; anomaly score = distance to the closest clean (bank) embeddings
        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=str(device))
        self.model = self.pipeline.model  # register the underlying nn.Module (parameter count, .to(), ...)

    def forward(self, x):

        # Input:

        # x: bs x window_size x nvars, must be on cpu (the pipeline moves data to the model device itself)
        bs, win_size, nvars = x.shape
        # Chronos-2 is natively multivariate: (bs, nvars, history), information is shared across variates
        ctx = rearrange(x, "bs win_size nvars -> bs nvars win_size")
        embs, _ = self.pipeline.embed(ctx)  # list of bs tensors: nvars x (n_patches+2) x d_model
        emb = torch.stack(embs, dim=0)  # bs x nvars x (n_patches+2) x d_model
        emb = emb[:, :, -1, :]  # last token = masked output patch (the forecast query, summarizes the context)
        emb = rearrange(emb, "bs nvars emb_dim -> bs (nvars emb_dim)")
        return emb.to(self.model.device)



class Chronos2BankEmbeddingCosine(BaseDetector):
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

        self.model = Chronos2(window_size=win_size, n_vars=feats, device=self.device)

        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def fit(self, data):
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        train_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        # device=None: batches stay on cpu, the Chronos-2 pipeline handles the device transfer itself
        bank_embedding = get_bank_embedding(self.model, train_loader, num_cores=self.num_cores, metric=self.metric, device=None)
        self.bank_embedding = bank_embedding.to(self.device)

    def decision_function(self, data):
        data = self.scaler.transform(data)

        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        scores = compute_anomaly_score(self.model, test_loader, self.bank_embedding,
                                       metric=self.metric, top_k=self.top_k, device=None)

        if scores.shape[0] < len(data):
            scores = np.array([scores[0]]*math.ceil((self.win_size-1)/2) +
                        list(scores) + [scores[-1]]*((self.win_size-1)//2))

        self.__anomaly_score = scores
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
