import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "eval", "metrics", "uni")
FIGS_DIR = os.path.join(HERE, "figs")
METRICS = ["VUS-ROC", "VUS-PR"]

os.makedirs(FIGS_DIR, exist_ok=True)

CATEGORY_COLORS = {
    "Forecaster": "#D55E00",   
    "Bank": "#56B4E9",         
    "Bank+PAI": "#0072B2",      
    "Var+SqDiff": "#555555",   
    "KNN": "#E69F00",          
    "Bank+kNN": "#009E73",     
    "Bank+PAI+kNN": "#006B57", 
}
CATEGORY_ORDER = ["Forecaster", "Bank", "Bank+PAI", "Var+SqDiff", "KNN", "Bank+kNN", "Bank+PAI+kNN"]

BACKBONES = ["TiRex", "FlowState", "T0alpha", "Chronos2", "Toto2"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_all_results(results_dir=RESULTS_DIR):
    df_metrics = {metric: pd.DataFrame() for metric in METRICS}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv"):
            continue
        name = fname[: -len(".csv")]
        res = pd.read_csv(os.path.join(results_dir, fname))
        for metric in METRICS:
            df_metrics[metric][name] = res[metric]
    return df_metrics


def aligned(df_metrics, metric):
    return df_metrics[metric].dropna(axis=1, how="all").dropna(axis=0, how="any")


def avg_rank(df):
    return df.rank(axis=1, ascending=False).mean().sort_values(ascending=True)


def categorize(name):
    if name.endswith("Forecaster"):
        return "Forecaster"
    if name == "KNN":
        return "KNN"
    if name == "OneLiner":
        return "Var+SqDiff"
    if name.startswith("KNN") and name.endswith("PAIFusion"):
        return "Bank+PAI+kNN"
    if name.startswith("KNN") and name.endswith("Fusion"):
        return "Bank+kNN"
    if name.endswith("BankEmbeddingPAI"):
        return "Bank+PAI"
    if name.endswith("BankEmbeddingCosine") or name.endswith("BankEmbeddingEuclidean"):
        return "Bank"
    return "Other"


def category_score_matrix(df):
    cat_of = {name: categorize(name) for name in df.columns}
    out = pd.DataFrame(index=df.index)
    for category in CATEGORY_ORDER:
        members = [name for name, c in cat_of.items() if c == category]
        if members:
            out[category] = df[members].mean(axis=1)
    return out


def backbone_of(name):
    for backbone in BACKBONES:
        if backbone in name:
            return backbone
    return None


def to_markdown_table(df, float_fmt="{:.3f}"):

    headers = ["index"] + list(df.columns)

    def fmt(v):
        return float_fmt.format(v) if isinstance(v, float) else str(v)

    rows = [[str(idx)] + [fmt(v) for v in row] for idx, row in zip(df.index, df.values)]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "|-" + "-|-".join("-" * w for w in widths) + "-|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    return "\n".join(lines)


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"{name}.{ext}"))
    print(f"saved {os.path.join(FIGS_DIR, name)}.{{pdf,png}}")
