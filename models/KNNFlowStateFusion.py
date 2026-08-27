from models.KNNBankFusion import KNNBankFusion
from models.FlowStateBankEmbeddingCosine import FlowState


class KNNFlowStateFusion(KNNBankFusion):
    def __init__(self, scale_factor=1.0, **kwargs):
        self.scale_factor = scale_factor
        super().__init__(**kwargs)

    def _build_model(self):
        return FlowState(window_size=self.win_size, n_vars=self.feats, device=self.device,
                         scale_factor=self.scale_factor).to(self.device)
