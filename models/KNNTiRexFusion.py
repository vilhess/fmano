from models.KNNBankFusion import KNNBankFusion
from models.TiRexBankEmbeddingCosine import TiRex


class KNNTiRexFusion(KNNBankFusion):
    def _build_model(self):
        return TiRex(window_size=self.win_size, n_vars=self.feats, device=self.device).to(self.device)
