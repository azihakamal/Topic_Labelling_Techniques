#!/usr/bin/env python3
"""
Combine all processed CSVs and visualize label distribution,
ignoring error labels in charts but keeping them in the combined CSV.
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
#from wordcloud import WordCloud

# ==============================
#  CONFIGURATION
# ==============================
OUTPUT_DIR = Path("./label_article/newspaper_articles/output_articles_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
#  COMBINE CSVs
# ==============================
csv_files = list(OUTPUT_DIR.glob("*.csv"))
all_dfs = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        all_dfs.append(df)
    except Exception as e:
        print(f"[WARN] Could not read {file}: {e}")

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined {len(csv_files)} CSVs → {len(combined_df)} rows")

# Save combined dataset (with error rows kept)
combined_csv_path = OUTPUT_DIR / "all_articles_with_labels.csv"
combined_df.to_csv(combined_csv_path, index=False)
print(f"Saved → {combined_csv_path}")

# ==============================
#  FILTER ONLY FOR VISUALIZATION
# ==============================
filtered_df = combined_df[combined_df["llm_label"] != "[ERROR] Failed to get label"]

# Recompute label counts without errors
label_counts = filtered_df["llm_label"].value_counts()

# Save label distribution without errors
label_counts_path = OUTPUT_DIR / "label_distribution_filtered.csv"
label_counts.to_csv(label_counts_path)
print(f"Saved → {label_counts_path}")

# ==============================
#  BAR CHART (Top 20, Errors Removed)
# ==============================
plt.figure(figsize=(10, 6))
label_counts.head(20).plot(kind="bar")
plt.title("Top 20 Thematic Labels")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

bar_chart_path = OUTPUT_DIR / "label_distribution_top20_filtered.png"
plt.savefig(bar_chart_path)
plt.show()
print(f"Saved → {bar_chart_path}")

# ==============================
#  WORD CLOUD (Errors Removed)
# ==============================
"""text = " ".join(filtered_df["llm_label"].dropna().astype(str))

wordcloud = WordCloud(
    width=1000, height=600,
    background_color="white",
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
).generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Labels")

wordcloud_path = OUTPUT_DIR / "label_distribution_wordcloud_filtered.png"
plt.savefig(wordcloud_path)
plt.show()
print(f"Saved → {wordcloud_path}")"""
