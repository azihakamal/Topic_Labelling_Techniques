# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
import plotly.io as pio

# --- FIX for custom vectorizer functions ---
def identity_func(x):
    return x

# --- SETUP ---
pio.templates.default = "plotly_white"

# Ensure plotly can export PNG
# Run this ONCE in your environment if not installed:
# pip install -U kaleido

# --- PATHS ---
OUTPUT_DIR = Path("./label_article/sample_articles/output_article_labeling_new")
MODEL_PATH = OUTPUT_DIR / "bertopic_model"
DATA_PATH = OUTPUT_DIR / "all_articles_with_topics.csv"

# --- LOAD MODEL AND DATA ---
print(" Loading model and topic data")
topic_model = BERTopic.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
print(f" Loaded {len(df)} articles")

# --- VISUALIZATION DIRECTORY ---
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ---  BAR CHART ---
print("Creating bar chart")
fig_bar = topic_model.visualize_barchart(top_n_topics=20)
fig_bar.write_html(VIZ_DIR / "viz_barchart.html")
fig_bar.write_image(VIZ_DIR / "viz_barchart.png", scale=2)
print(" Saved barchart PNG & HTML")

# --- TOPIC MAP ---
print("Creating topic map")
fig_topics = topic_model.visualize_topics()
fig_topics.write_html(VIZ_DIR / "viz_topics.html")
fig_topics.write_image(VIZ_DIR / "viz_topics.png", scale=2)
print("Saved topic map PNG & HTML")

# --- TOPIC HIERARCHY ---
print(" Creating topic hierarchy")
fig_hierarchy = topic_model.visualize_hierarchy()
fig_hierarchy.write_html(VIZ_DIR / "viz_hierarchy.html")
fig_hierarchy.write_image(VIZ_DIR / "viz_hierarchy.png", scale=2)
print("Saved hierarchy PNG & HTML")

# --- SIMILARITY HEATMAP ---
print(" Creating similarity heatmap")
fig_heatmap = topic_model.visualize_heatmap()
fig_heatmap.write_html(VIZ_DIR / "viz_heatmap.html")
fig_heatmap.write_image(VIZ_DIR / "viz_heatmap.png", scale=2)
print("Saved heatmap PNG & HTML")

print("\n All visualizations saved in:", VIZ_DIR)
