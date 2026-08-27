from models.KNNBankFusion import KNNBankFusion
from models.Chronos2BankEmbeddingCosine import Chronos2


class KNNChronos2Fusion(KNNBankFusion):
    # the Chronos-2 pipeline expects cpu batches and handles the device transfer itself
    _encode_on_cpu = True

    def _build_model(self):
        return Chronos2(window_size=self.win_size, n_vars=self.feats, device=self.device)
