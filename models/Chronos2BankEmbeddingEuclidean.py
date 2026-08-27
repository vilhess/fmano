from models.Chronos2BankEmbeddingCosine import Chronos2BankEmbeddingCosine


class Chronos2BankEmbeddingEuclidean(Chronos2BankEmbeddingCosine):
    # pure euclidean bank on unnormalized embeddings (the embedding norm may carry
    # amplitude information); equals the lambda_g = lambda_q = 0 cell of the PAI variant
    def __init__(self, **kwargs):
        super().__init__(metric='euclidean', **kwargs)
