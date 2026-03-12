import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

# ======================
# Configuration
# ======================
CSV_PATH = "./label_article/final_label/combine_csv/unseen_bertopic_fixed_categories_topk5.csv"
TOP_N = 10

OUT_DIR = Path("label_article/final_label/figure")
OUT_DIR.mkdir(exist_ok=True)

COLS = {
    "Supervised Classification": "pred_labels",
    "Zero-shot Classification": "llm_labels",
    "Hybrid (BERTopic + LLM)": "bertopic_final_categories_joined",
}

# Light pastel palette (10 distinct light colors)
PASTEL_10 = [
    "#c6dbef", "#d9f0a3", "#fdd0a2", "#dadaeb", "#fcbba1",
    "#c7e9c0", "#f2f0f7", "#fde0dd", "#e5f5e0", "#fee6ce"
]

# ======================
# Helper functions
# ======================
def split_labels(x):
    if pd.isna(x):
        return []
    parts = re.split(r"\s*[;,|]\s*", str(x))
    return [p.strip() for p in parts if p.strip()]

def label_counts(series):
    return series.apply(split_labels).explode().value_counts()

# ======================
# Load data
# ======================
df = pd.read_csv(CSV_PATH)

# ======================
# Plot per technique
# ======================
for method, col in COLS.items():
    counts = label_counts(df[col]).head(TOP_N)

    # Ensure exactly TOP_N colors (or fewer if < TOP_N labels)
    colors = PASTEL_10[:len(counts)]

    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values, color=colors)

    plt.title(f"Top 10 Label Distribution — {method}")
    plt.xlabel("Label")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    fname = (
        method.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )

    plt.savefig(OUT_DIR / f"{fname}_top10_multicolor.png", dpi=300)
    #plt.savefig(OUT_DIR / f"{fname}_top10_multicolor.svg")
    plt.close()

print(f"Saved plots to: {OUT_DIR.resolve()}")