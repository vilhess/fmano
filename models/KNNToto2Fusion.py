from models.KNNBankFusion import KNNBankFusion
from models.Toto2BankEmbeddingCosine import Toto2


class KNNToto2Fusion(KNNBankFusion):
    def _build_model(self):
        return Toto2(window_size=self.win_size, n_vars=self.feats, device=self.device)
