import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File paths ===
bert_path = "./label_article/newspaper_articles/output_articles_BERT/all_articles_BERT_combined.csv"
bertopic_path = "./label_article/newspaper_articles/output_article_labeling_final/all_articles_with_topics.csv"
llm_path = "./label_article/newspaper_articles/output_articles_llm/all_articles_with_labels.csv"

# ===  Column names ===
bert_col = "BERT_label"
bertopic_col = "topic_label"
llm_col = "llm_label"

# ===  Output directory ===
output_dir = "./label_article/newspaper_articles/visualization"
os.makedirs(output_dir, exist_ok=True)

# === Load CSVs ===
df_bert = pd.read_csv(bert_path)
df_bertopic = pd.read_csv(bertopic_path)
df_llm = pd.read_csv(llm_path)

# === Top 9 topics per method ===
top_n = 9
bert_counts = df_bert[bert_col].value_counts().head(top_n)
bertopic_counts = df_bertopic[bertopic_col].value_counts().head(top_n)
llm_counts = df_llm[llm_col].value_counts().head(top_n)

# === Plot style settings ===
sns.set(style="whitegrid", font="Helvetica", font_scale=1.0)

# Helper function to plot and save
def plot_and_save(counts, title, palette, filename):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        palette=palette
    )
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Article Count", fontsize=13)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {filename}")

# === Plot each separately ===
plot_and_save(
    bert_counts,
    "BERT – Top 9 Topics",
    sns.color_palette("Blues_r", top_n),
    "bert_bar_chart_top9.png"
)

plot_and_save(
    bertopic_counts,
    "BERTopic – Top 9 Topics",
    sns.color_palette("Greens_r", top_n),
    "bertopic_bar_chart_top9.png"
)

plot_and_save(
    llm_counts,
    "LLM – Top 9 Topics",
    sns.color_palette("Purples_r", top_n),
    "llm_bar_chart_top9.png"
)
