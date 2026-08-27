def run_TiRexForecaster(data_train, data, win_size=10):
    from models.TiRexForecaster import TiRexForecaster
    clf = TiRexForecaster(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_TiRexBankEmbeddingCosine(data_train, data, win_size=10):
    from models.TiRexBankEmbeddingCosine import TiRexBankEmbeddingCosine
    clf = TiRexBankEmbeddingCosine(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_FlowStateForecaster(data_train, data, win_size=10, scale_factor=1.0):
    from models.FlowStateForecaster import FlowStateForecaster
    clf = FlowStateForecaster(win_size=win_size, feats=data.shape[1], scale_factor=scale_factor)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_FlowStateBankEmbeddingCosine(data_train, data, win_size=10, scale_factor=1.0):
    from models.FlowStateBankEmbeddingCosine import FlowStateBankEmbeddingCosine
    clf = FlowStateBankEmbeddingCosine(win_size=win_size, feats=data.shape[1], scale_factor=scale_factor)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_T0alphaForecaster(data_train, data, win_size=10):
    from models.T0alphaForecaster import T0alphaForecaster
    clf = T0alphaForecaster(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_T0alphaBankEmbeddingCosine(data_train, data, win_size=10):
    from models.T0alphaBankEmbeddingCosine import T0alphaBankEmbeddingCosine
    clf = T0alphaBankEmbeddingCosine(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Chronos2Forecaster(data_train, data, win_size=10):
    from models.Chronos2Forecaster import Chronos2Forecaster
    clf = Chronos2Forecaster(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Chronos2BankEmbeddingCosine(data_train, data, win_size=10):
    from models.Chronos2BankEmbeddingCosine import Chronos2BankEmbeddingCosine
    clf = Chronos2BankEmbeddingCosine(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Toto2Forecaster(data_train, data, win_size=10):
    from models.Toto2Forecaster import Toto2Forecaster
    clf = Toto2Forecaster(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Toto2BankEmbeddingCosine(data_train, data, win_size=10):
    from models.Toto2BankEmbeddingCosine import Toto2BankEmbeddingCosine
    clf = Toto2BankEmbeddingCosine(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_TiRexBankEmbeddingEuclidean(data_train, data, win_size=10):
    from models.TiRexBankEmbeddingEuclidean import TiRexBankEmbeddingEuclidean
    clf = TiRexBankEmbeddingEuclidean(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_FlowStateBankEmbeddingEuclidean(data_train, data, win_size=10, scale_factor=1.0):
    from models.FlowStateBankEmbeddingEuclidean import FlowStateBankEmbeddingEuclidean
    clf = FlowStateBankEmbeddingEuclidean(win_size=win_size, feats=data.shape[1], scale_factor=scale_factor)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_T0alphaBankEmbeddingEuclidean(data_train, data, win_size=10):
    from models.T0alphaBankEmbeddingEuclidean import T0alphaBankEmbeddingEuclidean
    clf = T0alphaBankEmbeddingEuclidean(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Chronos2BankEmbeddingEuclidean(data_train, data, win_size=10):
    from models.Chronos2BankEmbeddingEuclidean import Chronos2BankEmbeddingEuclidean
    clf = Chronos2BankEmbeddingEuclidean(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Toto2BankEmbeddingEuclidean(data_train, data, win_size=10):
    from models.Toto2BankEmbeddingEuclidean import Toto2BankEmbeddingEuclidean
    clf = Toto2BankEmbeddingEuclidean(win_size=win_size, feats=data.shape[1])
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_TiRexBankEmbeddingPAI(data_train, data, win_size=10, lambda_g=0.667, lambda_q=0.2):
    from models.TiRexBankEmbeddingPAI import TiRexBankEmbeddingPAI
    clf = TiRexBankEmbeddingPAI(win_size=win_size, feats=data.shape[1], lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_FlowStateBankEmbeddingPAI(data_train, data, win_size=10, scale_factor=1.0, lambda_g=0.667, lambda_q=0.2):
    from models.FlowStateBankEmbeddingPAI import FlowStateBankEmbeddingPAI
    clf = FlowStateBankEmbeddingPAI(win_size=win_size, feats=data.shape[1], scale_factor=scale_factor,
                                    lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_T0alphaBankEmbeddingPAI(data_train, data, win_size=10, lambda_g=0.667, lambda_q=0.2):
    from models.T0alphaBankEmbeddingPAI import T0alphaBankEmbeddingPAI
    clf = T0alphaBankEmbeddingPAI(win_size=win_size, feats=data.shape[1], lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Chronos2BankEmbeddingPAI(data_train, data, win_size=10, lambda_g=0.667, lambda_q=0.2):
    from models.Chronos2BankEmbeddingPAI import Chronos2BankEmbeddingPAI
    clf = Chronos2BankEmbeddingPAI(win_size=win_size, feats=data.shape[1], lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_Toto2BankEmbeddingPAI(data_train, data, win_size=10, lambda_g=0.667, lambda_q=0.2):
    from models.Toto2BankEmbeddingPAI import Toto2BankEmbeddingPAI
    clf = Toto2BankEmbeddingPAI(win_size=win_size, feats=data.shape[1], lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNN(data_train, data, win_size=100, n_neighbors=10, method='largest'):
    from models.KNN import KNN
    clf = KNN(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors, method=method)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()
    
def run_KNNT0alphaFusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0):
    from models.KNNT0alphaFusion import KNNT0alphaFusion
    clf = KNNT0alphaFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                           method=method, lambda_bank=lambda_bank)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNTiRexFusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0):
    from models.KNNTiRexFusion import KNNTiRexFusion
    clf = KNNTiRexFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                         method=method, lambda_bank=lambda_bank)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNFlowStateFusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0, scale_factor=1.0):
    from models.KNNFlowStateFusion import KNNFlowStateFusion
    clf = KNNFlowStateFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                             method=method, lambda_bank=lambda_bank, scale_factor=scale_factor)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNChronos2Fusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0):
    from models.KNNChronos2Fusion import KNNChronos2Fusion
    clf = KNNChronos2Fusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                            method=method, lambda_bank=lambda_bank)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNToto2Fusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0):
    from models.KNNToto2Fusion import KNNToto2Fusion
    clf = KNNToto2Fusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                         method=method, lambda_bank=lambda_bank)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNTiRexPAIFusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0, lambda_g=0.667, lambda_q=0.2):
    from models.KNNTiRexPAIFusion import KNNTiRexPAIFusion
    clf = KNNTiRexPAIFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                            method=method, lambda_bank=lambda_bank, lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNFlowStatePAIFusion(data_train, data, win_size=128, n_neighbors=50, method='mean', lambda_bank=1.0,
                              scale_factor=1.0, lambda_g=0.667, lambda_q=0.0):
    from models.KNNFlowStatePAIFusion import KNNFlowStatePAIFusion
    clf = KNNFlowStatePAIFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors, method=method,
                                lambda_bank=lambda_bank, scale_factor=scale_factor, lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNT0alphaPAIFusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0, lambda_g=0.667, lambda_q=0.2):
    from models.KNNT0alphaPAIFusion import KNNT0alphaPAIFusion
    clf = KNNT0alphaPAIFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                              method=method, lambda_bank=lambda_bank, lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNChronos2PAIFusion(data_train, data, win_size=128, n_neighbors=50, method='mean', lambda_bank=1.0, lambda_g=0.667, lambda_q=0.2):
    from models.KNNChronos2PAIFusion import KNNChronos2PAIFusion
    clf = KNNChronos2PAIFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                               method=method, lambda_bank=lambda_bank, lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_KNNToto2PAIFusion(data_train, data, win_size=64, n_neighbors=50, method='mean', lambda_bank=1.0, lambda_g=0.667, lambda_q=0.2):
    from models.KNNToto2PAIFusion import KNNToto2PAIFusion
    clf = KNNToto2PAIFusion(win_size=win_size, feats=data.shape[1], n_neighbors=n_neighbors,
                            method=method, lambda_bank=lambda_bank, lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_AmplitudeOnly(data_train, data, lambda_g=0.667, lambda_q=0.2):
    from models.AmplitudeOnly import AmplitudeOnly
    clf = AmplitudeOnly(feats=data.shape[1], lambda_g=lambda_g, lambda_q=lambda_q)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

def run_OneLiner(data_train, data, lambda_mw=0.667, lambda_sq=0.2, window_mw=32, window_sq=2):
    from models.OneLiner import OneLiner
    clf = OneLiner(feats=data.shape[1], lambda_mw=lambda_mw, lambda_sq=lambda_sq,
                    window_mw=window_mw, window_sq=window_sq)
    clf.fit(data_train)
    score = clf.decision_function(data)
    return score.ravel()

AD_Pool = ['TiRexForecaster', 'TiRexBankEmbeddingCosine', 'FlowStateForecaster', 'FlowStateBankEmbeddingCosine', 'T0alphaForecaster', 'T0alphaBankEmbeddingCosine', 'Chronos2Forecaster', 'Chronos2BankEmbeddingCosine', 'Toto2Forecaster', 'Toto2BankEmbeddingCosine', 'TiRexBankEmbeddingEuclidean', 'FlowStateBankEmbeddingEuclidean', 'T0alphaBankEmbeddingEuclidean', 'Chronos2BankEmbeddingEuclidean', 'Toto2BankEmbeddingEuclidean', 'TiRexBankEmbeddingPAI', 'FlowStateBankEmbeddingPAI', 'T0alphaBankEmbeddingPAI', 'Chronos2BankEmbeddingPAI', 'Toto2BankEmbeddingPAI', 'KNN', 'KNNT0alphaFusion', 'KNNTiRexFusion', 'KNNFlowStateFusion', 'KNNChronos2Fusion', 'KNNToto2Fusion', 'KNNTiRexPAIFusion', 'KNNFlowStatePAIFusion', 'KNNT0alphaPAIFusion', 'KNNChronos2PAIFusion', 'KNNToto2PAIFusion', 'AmplitudeOnly', 'OneLiner']


def run_AD(AD_Name, data_train, data, **kwargs):
    if AD_Name not in AD_Pool:
        raise ValueError(f"Unknown AD model: {AD_Name}. Available models: {AD_Pool}")
    function_to_call = globals()[f'run_{AD_Name}']
    return function_to_call(data_train, data, **kwargs)
