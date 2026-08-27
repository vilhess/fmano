from models.KNNBankFusion import KNNBankFusion
from models.T0alphaBankEmbeddingCosine import T0alpha


class KNNT0alphaFusion(KNNBankFusion):
    def _build_model(self):
        return T0alpha(window_size=self.win_size, n_vars=self.feats, device=self.device).to(self.device)
