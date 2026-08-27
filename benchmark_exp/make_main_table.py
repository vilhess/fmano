import os
import pandas as pd

from figs_common import (
    FIGS_DIR, METRICS, load_all_results, aligned, avg_rank, categorize, CATEGORY_ORDER,
    category_score_matrix, to_markdown_table,
)


def build_full_table(df_metrics):
    ranks = {metric: avg_rank(aligned(df_metrics, metric)) for metric in METRICS}
    means = {metric: aligned(df_metrics, metric).mean() for metric in METRICS}

    all_names = ranks[METRICS[0]].index
    table = pd.DataFrame(index=all_names)
    table["category"] = [categorize(n) for n in all_names]
    for metric in METRICS:
        table[f"{metric} (rank)"] = ranks[metric].reindex(all_names)
        table[f"{metric} (mean)"] = means[metric].reindex(all_names)

    table = table.sort_values(f"{METRICS[0]} (rank)")
    return table


def build_category_table(df_metrics, full_table):
    pooled = full_table

    means = pooled.groupby("category").agg(
        {f"{metric} (mean)": "mean" for metric in METRICS}
    )
    n_detectors = pooled.groupby("category").size()

    agg = means.copy()
    for metric in METRICS:
        df_cat = category_score_matrix(aligned(df_metrics, metric))
        agg[f"{metric} (rank)"] = df_cat.rank(axis=1, ascending=False).mean()
    agg["n_detectors"] = n_detectors

    column_order = [c for metric in METRICS for c in (f"{metric} (rank)", f"{metric} (mean)")] + ["n_detectors"]
    agg = agg[column_order]
    agg = agg.reindex([c for c in CATEGORY_ORDER if c in agg.index])
    agg = agg.sort_values(f"{METRICS[0]} (rank)")
    return agg


def save(df, basename, tex=False):
    df.to_csv(os.path.join(FIGS_DIR, f"{basename}.csv"), float_format="%.3f")
    with open(os.path.join(FIGS_DIR, f"{basename}.md"), "w") as f:
        f.write(to_markdown_table(df))
    if tex:
        with open(os.path.join(FIGS_DIR, f"{basename}.tex"), "w") as f:
            f.write(df.to_latex(float_format="%.3f"))
    print(f"saved {os.path.join(FIGS_DIR, basename)}.{{csv,md{',tex' if tex else ''}}}")


if __name__ == "__main__":
    df_metrics = load_all_results()

    full_table = build_full_table(df_metrics)
    save(full_table, "table_full_detectors")

    category_table = build_category_table(df_metrics, full_table)
    save(category_table, "table_main_categories", tex=True)

    print("\n=== Condensed category table (Table 4.1) ===")
    print(category_table.to_string(float_format=lambda x: f"{x:.3f}"))
