import matplotlib.pyplot as plt
from aeon.visualisation import plot_critical_difference

from figs_common import METRICS, CATEGORY_COLORS, load_all_results, aligned, category_score_matrix, savefig

if __name__ == "__main__":
    df_metrics = load_all_results()

    for metric in METRICS:
        df = aligned(df_metrics, metric)
        df_cat = category_score_matrix(df)

        highlight = {c: CATEGORY_COLORS[c] for c in df_cat.columns}
        fig, ax = plot_critical_difference(df_cat.values, df_cat.columns, highlight=highlight)
        ax.set_title(f"Critical Difference Diagram by family -- {metric}")
        savefig(fig, f"fig_cd_categories_{metric.lower().replace('-', '_')}")
        plt.close(fig)
