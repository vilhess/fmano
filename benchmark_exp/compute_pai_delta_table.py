import pandas as pd

from figs_common import METRICS, BACKBONES, load_all_results, aligned

DISPLAY_NAME = {
    "TiRex": "TiRex",
    "FlowState": "FlowState",
    "T0alpha": "T0-alpha",
    "Chronos2": "Chronos-2",
    "Toto2": "Toto-2",
}

BANK_EUCL = {b: f"{b}BankEmbeddingEuclidean" for b in BACKBONES}
BANK_PAI = {b: f"{b}BankEmbeddingPAI" for b in BACKBONES}
FUSION = {b: f"KNN{b}Fusion" for b in BACKBONES}
FUSION_PAI = {b: f"KNN{b}PAIFusion" for b in BACKBONES}


def compute_deltas(df_metrics):
    rows = []
    for backbone in BACKBONES:
        row = {"Backbone": backbone}
        for metric in METRICS:
            df = aligned(df_metrics, metric)
            alone = df[BANK_PAI[backbone]] - df[BANK_EUCL[backbone]]
            in_fusion = df[FUSION_PAI[backbone]] - df[FUSION[backbone]]
            row[(metric, "alone")] = alone.mean()
            row[(metric, "in fusion")] = in_fusion.mean()
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Backbone")
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    table.loc["Mean"] = table.mean()
    return table


if __name__ == "__main__":
    df_metrics = load_all_results()

    table = compute_deltas(df_metrics)
    table.index = [DISPLAY_NAME.get(b, b) for b in table.index]

    print(table.to_string(float_format=lambda x: f"{x:+.3f}"))
