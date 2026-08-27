import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from TSB_AD.base import BaseDetector
from TSB_AD.utils.dataset import ReconstructDataset
from TSB_AD.utils.torch_utility import get_gpu

from models.bank_scoring import get_bank_embedding, compute_anomaly_score

from toto2 import Toto2Model
from einops import rearrange

class Toto2(nn.Module):
    def __init__(self, window_size, n_vars, device):
        super().__init__()
        # embeds the window; anomaly score = distance to the closest clean (bank) embeddings
        self.model = Toto2Model.from_pretrained("Datadog/Toto-2.0-22m").to(device).eval()

    def forward(self, x):

        # Input:

        # x: bs x window_size x nvars, on the model device
        bs, win_size, nvars = x.shape
        # Toto2 is natively multivariate: (bs, nvars, history), information is shared across variates
        ctx = rearrange(x, "bs win_size nvars -> bs nvars win_size")
        mask = torch.ones((bs, nvars, win_size), dtype=torch.bool, device=x.device)

        # left-pad with unobserved values: the context length must be a multiple of patch_size
        patch_size = self.model.config.patch_size
        pad = (-win_size) % patch_size
        if pad > 0:
            ctx = F.pad(ctx, (pad, 0))
            mask = F.pad(mask, (pad, 0), value=False)

        ids = torch.zeros((bs, nvars), dtype=torch.long, device=x.device)

        # capture the transformer output (bs x nvars x n_patches x d_model) with a forward hook
        captured = {}
        hook = self.model.transformer.register_forward_hook(
            lambda module, args, output: captured.__setitem__("x", output)
        )
        try:
            self.model(target=ctx, target_mask=mask, cpm_mask=mask, series_ids=ids, num_return_steps=1)
        finally:
            hook.remove()

        emb = captured["x"][..., -1, :]  # last patch token (causal transformer, summarizes the window): bs x nvars x d_model
        emb = rearrange(emb, "bs nvars emb_dim -> bs (nvars emb_dim)")
        return emb



class Toto2BankEmbeddingCosine(BaseDetector):
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

        self.model = Toto2(window_size=win_size, n_vars=feats, device=self.device)

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
