from models.KNNDetectorFusion import KNNDetectorFusion
from models.FlowStateBankEmbeddingPAI import FlowStateBankEmbeddingPAI

# KNN + FlowState bank fused with PAI amplitude channels. See
# models/KNNDetectorFusion.py for the fusion formula. FlowState+PAI component frozen
# at its own tuned optimum; win_size is shared between the KNN component and the
# wrapped detector.


class KNNFlowStatePAIFusion(KNNDetectorFusion):
    def __init__(self,
                 batch_size = 256,
                 num_cores = 500,
                 top_k = 3,
                 scale_factor = 1.0,
                 w_b = 1.0,
                 lambda_g = 0.667,
                 lambda_q = 0.2,
                 pai_window = 32,
                 **kwargs):
        self._batch_size = batch_size
        self._num_cores = num_cores
        self._top_k = top_k
        self._scale_factor = scale_factor
        self._w_b = w_b
        self._lambda_g = lambda_g
        self._lambda_q = lambda_q
        self._pai_window = pai_window
        super().__init__(**kwargs)

    def _build_detector(self):
        return FlowStateBankEmbeddingPAI(
            win_size=self.win_size, feats=self.feats, batch_size=self._batch_size,
            num_cores=self._num_cores, top_k=self._top_k, scale_factor=self._scale_factor,
            w_b=self._w_b, lambda_g=self._lambda_g, lambda_q=self._lambda_q, pai_window=self._pai_window,
        )
