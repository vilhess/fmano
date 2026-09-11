import numpy as np

from figs_common import METRICS, load_all_results, aligned

BACKBONES = ["TiRex", "FlowState", "T0alpha", "Chronos2", "Toto2"]


def per_backbone_correlation(df):
    """corr(KNN, Bank_eucl), corr(KNN, Bank+PAI) per backbone, then mean/std across backbones."""
    corrs_eucl, corrs_pai = [], []
    for b in BACKBONES:
        corrs_eucl.append(df["KNN"].corr(df[f"{b}BankEmbeddingEuclidean"], method="spearman"))
        corrs_pai.append(df["KNN"].corr(df[f"{b}BankEmbeddingPAI"], method="spearman"))
    corrs_eucl, corrs_pai = np.array(corrs_eucl), np.array(corrs_pai)
    return corrs_eucl.mean(), corrs_eucl.std(), corrs_pai.mean(), corrs_pai.std()


if __name__ == "__main__":
    df_metrics = load_all_results()

    print(f"{'':10s} {'corr(KNN, Bank_eucl)':>24s} {'corr(KNN, Bank+PAI)':>24s}")
    for metric in METRICS:
        df = aligned(df_metrics, metric)
        mean_eucl, std_eucl, mean_pai, std_pai = per_backbone_correlation(df)
        print(
            f"{metric:10s} {mean_eucl:10.2f} ± {std_eucl:<9.2f} "
            f"{mean_pai:10.2f} ± {std_pai:<9.2f}"
        )
