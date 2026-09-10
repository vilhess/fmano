import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figs_common import METRICS, CATEGORY_COLORS, HERE, load_all_results, aligned, savefig

BASELINE_DIR = os.path.join(HERE, "eval", "metrics", "uni_baseline")
MATRIXPROFILE_COLOR = "#CC79A7" 


def load_baseline(name):
    return pd.read_csv(os.path.join(BASELINE_DIR, f"{name}.csv"))

BACKBONE_ORDER = ["TiRex", "FlowState", "T0alpha", "Chronos2", "Toto2"]
DISPLAY_NAME = {"TiRex": "TiRex", "FlowState": "FlowState", "T0alpha": "T0-alpha",
                "Chronos2": "Chronos-2", "Toto2": "Toto-2"}
TICK_FONTSIZE = 11  
LABEL_FONTSIZE = 12 
LEGEND_FONTSIZE = 11
LEGEND_MARKERSIZE = 9
DOT_SIZE = 80          
DIAMOND_SIZE = 75      
ROW_SPACING = 0.7      
REF_LINE_WIDTH = 2.6   

DETECTORS = {
    "TiRex": [
        ("TiRexForecaster", "Forecaster", "o"),
        ("TiRexBankEmbeddingCosine", "Bank", "o"),
        ("TiRexBankEmbeddingEuclidean", "Bank", "D"),
        ("TiRexBankEmbeddingPAI", "Bank+PAI", "o"),
        ("KNNTiRexFusion", "Bank+kNN", "o"),
        ("KNNTiRexPAIFusion", "Bank+PAI+kNN", "o"),
    ],
    "FlowState": [
        ("FlowStateForecaster", "Forecaster", "o"),
        ("FlowStateBankEmbeddingCosine", "Bank", "o"),
        ("FlowStateBankEmbeddingEuclidean", "Bank", "D"),
        ("FlowStateBankEmbeddingPAI", "Bank+PAI", "o"),
        ("KNNFlowStateFusion", "Bank+kNN", "o"),
        ("KNNFlowStatePAIFusion", "Bank+PAI+kNN", "o"),
    ],
    "T0alpha": [
        ("T0alphaForecaster", "Forecaster", "o"),
        ("T0alphaBankEmbeddingCosine", "Bank", "o"),
        ("T0alphaBankEmbeddingEuclidean", "Bank", "D"),
        ("T0alphaBankEmbeddingPAI", "Bank+PAI", "o"),
        ("KNNT0alphaFusion", "Bank+kNN", "o"),
        ("KNNT0alphaPAIFusion", "Bank+PAI+kNN", "o"),
    ],
    "Chronos2": [
        ("Chronos2Forecaster", "Forecaster", "o"),
        ("Chronos2BankEmbeddingCosine", "Bank", "o"),
        ("Chronos2BankEmbeddingEuclidean", "Bank", "D"),
        ("Chronos2BankEmbeddingPAI", "Bank+PAI", "o"),
        ("KNNChronos2Fusion", "Bank+kNN", "o"),
        ("KNNChronos2PAIFusion", "Bank+PAI+kNN", "o"),
    ],
    "Toto2": [
        ("Toto2Forecaster", "Forecaster", "o"),
        ("Toto2BankEmbeddingCosine", "Bank", "o"),
        ("Toto2BankEmbeddingEuclidean", "Bank", "D"),
        ("Toto2BankEmbeddingPAI", "Bank+PAI", "o"),
        ("KNNToto2Fusion", "Bank+kNN", "o"),
        ("KNNToto2PAIFusion", "Bank+PAI+kNN", "o"),
    ]
}
KNN_COLOR = CATEGORY_COLORS["KNN"]
ONELINER_COLOR = CATEGORY_COLORS["Var+SqDiff"]
LEGEND_CATEGORIES = ["Forecaster", "Bank", "Bank+PAI", "Bank+kNN", "Bank+PAI+kNN"]

LEGEND_LABELS = ["Forecaster", "Bank (cosine)", "Bank+PAI", "Bank+kNN", "Bank+PAI+kNN"]


def plot(df_metrics, df_matrixprofile):
    backbone_order = BACKBONE_ORDER

    fig, axes = plt.subplots(1, len(METRICS), figsize=(9.0, 3.1), sharey=True)

    for i, (ax, metric) in enumerate(zip(axes, METRICS)):
        df = aligned(df_metrics, metric)
        knn_score = df["KNN"].mean()
        oneliner_score = df["OneLiner"].mean()
        matrixprofile_score = df_matrixprofile.loc[df.index, metric].mean()

        y_positions = [i * ROW_SPACING for i in range(len(backbone_order))]
        for y, backbone in zip(y_positions, backbone_order):
            values = [df[name].mean() for name, _, _ in DETECTORS[backbone]]
            ax.plot([min(values), max(values)], [y, y], color="#CCCCCC", lw=1.7, zorder=1)
            for (name, category, marker), v in zip(DETECTORS[backbone], values):
                color = CATEGORY_COLORS[category]
                if marker == "D":
                    ax.scatter(v, y, marker=marker, s=DIAMOND_SIZE, zorder=2,
                               facecolors="white", edgecolors=color, linewidths=2.2)
                else:
                    ax.scatter(v, y, marker=marker, s=DOT_SIZE, zorder=2, color=color)

        ax.axvline(knn_score, color=KNN_COLOR, lw=REF_LINE_WIDTH, ls="--", zorder=0)
        ax.axvline(oneliner_score, color=ONELINER_COLOR, lw=REF_LINE_WIDTH, ls=":", zorder=0)
        ax.axvline(matrixprofile_score, color=MATRIXPROFILE_COLOR, lw=REF_LINE_WIDTH, ls="-.", zorder=0)

        ax.set_yticks(list(y_positions))

        if i == 0:
            ax.set_yticklabels([DISPLAY_NAME[b] for b in backbone_order], fontsize=TICK_FONTSIZE, fontweight="bold")
            ax.invert_yaxis()  
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(metric, fontsize=LABEL_FONTSIZE, pad=6, fontweight="bold")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=CATEGORY_COLORS[c], markersize=LEGEND_MARKERSIZE)
        for c in LEGEND_CATEGORIES
    ] + [
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white",
               markeredgecolor=CATEGORY_COLORS["Bank"], markeredgewidth=2.2, markersize=LEGEND_MARKERSIZE),
        Line2D([0], [0], color=KNN_COLOR, ls="--", lw=REF_LINE_WIDTH),
        Line2D([0], [0], color=ONELINER_COLOR, ls=":", lw=REF_LINE_WIDTH),
        Line2D([0], [0], color=MATRIXPROFILE_COLOR, ls="-.", lw=REF_LINE_WIDTH),
    ]
    labels = LEGEND_LABELS + [
        "Bank (euclidean)", "KNN alone", "Var+SqDiff floor", "MatrixProfile",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               prop={"size": LEGEND_FONTSIZE, "weight": "bold"},
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, labelspacing=0.4,
               handletextpad=0.5, handlelength=1.6)

    fig.tight_layout(rect=[0, 0.25, 1, 1])
    fig.subplots_adjust(wspace=0.35)
    return fig


if __name__ == "__main__":
    df_metrics = load_all_results()
    df_matrixprofile = load_baseline("MatrixProfile")

    fig = plot(df_metrics, df_matrixprofile)
    savefig(fig, "fig_generalization")
    plt.close(fig)
