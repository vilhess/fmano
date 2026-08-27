from models.Toto2BankEmbeddingCosine import Toto2BankEmbeddingCosine


class Toto2BankEmbeddingEuclidean(Toto2BankEmbeddingCosine):
    # pure euclidean bank on unnormalized embeddings (the embedding norm may carry
    # amplitude information); equals the lambda_g = lambda_q = 0 cell of the PAI variant
    def __init__(self, **kwargs):
        super().__init__(metric='euclidean', **kwargs)
