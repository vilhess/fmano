from figs_common import METRICS, load_all_results, aligned

BACKBONES = ["TiRex", "FlowState", "T0alpha", "Chronos2", "Toto2"]

BANK_EUCL = [f"{b}BankEmbeddingEuclidean" for b in BACKBONES]
BANK_PAI = [f"{b}BankEmbeddingPAI" for b in BACKBONES]


def pooled_correlation(df):
    """corr(KNN, pooled Bank_eucl), corr(KNN, pooled Bank+PAI) for one metric's df."""
    pooled_eucl = df[BANK_EUCL].mean(axis=1)
    pooled_pai = df[BANK_PAI].mean(axis=1)
    corr_eucl = df["KNN"].corr(pooled_eucl, method="spearman")
    corr_pai = df["KNN"].corr(pooled_pai, method="spearman")
    return corr_eucl, corr_pai


if __name__ == "__main__":
    df_metrics = load_all_results()

    print(f"{'':10s} {'corr(KNN, Bank_eucl)':>22s} {'corr(KNN, Bank+PAI)':>22s}")
    for metric in METRICS:
        df = aligned(df_metrics, metric)
        corr_eucl, corr_pai = pooled_correlation(df)
        print(f"{metric:10s} {corr_eucl:22.2f} {corr_pai:22.2f}")
