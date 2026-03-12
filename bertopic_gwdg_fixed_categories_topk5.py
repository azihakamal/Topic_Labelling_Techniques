# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# Needed to unpickle saved BERTopic vectorizer
def identity_func(x):
    return x


# ===================== PATHS =====================
CATEGORY_CSV = Path("./label_article/newspaper_articles/cropping_category.csv")
CATEGORY_COL = "name"

MODEL_DIR = Path("./label_article/newspaper_articles/output_article_labeling_final/bertopic_model")

GOLD_CSV = Path("./label_article/final_label/gold_label/test_predictions_topk.csv")
GOLD_TEXT_COL = "text"

UNSEEN_CSV = Path("./label_article/final_label/unseen_label/gwdg_combine_real_article_predictions_topk5.csv")
UNSEEN_TEXT_COL = "clean_article"

OUTPUT_DIR = Path("./label_article/final_label/combine_csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ===================== GWDG =====================
GWDG_BASE_URL = "GWDG_Link"
GWDG_API_KEY = os.getenv("GWDG_API_KEY", "key")
GWDG_MODEL = "llama-3.3-70b-instruct"

client = OpenAI(base_url=GWDG_BASE_URL, api_key=GWDG_API_KEY)

TOPIC_MAP_CACHE = OUTPUT_DIR / "topic_to_category_cache.json"

# Fallback category
FALLBACK_CATEGORY = "Allgemein"


# ===================== NORMALIZATION =====================
def norm_key(s: str) -> str:
    s = str(s).strip().casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


# ===================== TAXONOMY =====================
def load_categories(path: Path, col: str) -> List[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Category CSV missing column '{col}'. Columns: {list(df.columns)}")

    cats = df[col].dropna().astype(str).map(str.strip).tolist()

    seen = set()
    out = []
    for c in cats:
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    if not out:
        raise ValueError(f"No categories found in {path} column '{col}'.")
    print(f" Loaded {len(out)} categories from {path} (column '{col}').")
    return out


# ===================== CACHE =====================
def load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def save_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ===================== BERTopic TOP-K =====================
def add_bertopic_topk(df: pd.DataFrame, topic_model: BERTopic, text_col: str, top_k: int) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}'. Columns: {list(df.columns)}")

    df = df.copy()
    texts = df[text_col].fillna("").astype(str).tolist()

    doc_topics, probs = topic_model.transform(texts)
    if probs is None:
        raise RuntimeError(
            "topic_model.transform returned probs=None. "
            "Your saved BERTopic model cannot provide probability vectors, so top_k>1 cannot be computed.\n"
            "Fix: retrain BERTopic with calculate_probabilities=True, or set TOP_K=1."
        )

    topk_topic_ids = np.argsort(-probs, axis=1)[:, :top_k]
    topk_scores = np.take_along_axis(probs, topk_topic_ids, axis=1)

    df["bertopic_topic_id"] = doc_topics
    for i in range(top_k):
        df[f"bertopic_top{i+1}_topic_id"] = topk_topic_ids[:, i].astype(int)
        df[f"bertopic_top{i+1}_topic_score"] = topk_scores[:, i].astype(float)

    print(f"Computed BERTopic top_k={top_k} for '{text_col}' ({len(df)} rows).")
    return df


def topic_keywords_str(topic_model: BERTopic, topic_id: int, topn: int = 12) -> str:
    words = topic_model.get_topic(topic_id) or []
    return ", ".join([w for w, _ in words[:topn]])


def collect_topic_ids_from_bertopic_topk(df: pd.DataFrame, top_k: int) -> List[int]:
    tids: List[int] = []
    for i in range(1, top_k + 1):
        col = f"bertopic_top{i}_topic_id"
        tids.extend(df[col].dropna().astype(int).tolist())
    return sorted(set([t for t in tids if t != -1]))


# ===================== GWDG MAPPING =====================
def gwdg_map_topic_to_category(topic_id: int, keywords: str, categories: List[str]) -> Dict[str, str]:
    categories_block = "\n".join([f"- {c}" for c in categories])

    prompt = f"""
You map BERTopic topics to exactly ONE category from a fixed taxonomy.

Allowed categories (choose exactly one; MUST match one item exactly):
{categories_block}

Topic id: {topic_id}
Topic keywords: {keywords}

Rules:
- Output JSON only (no markdown, no commentary).
- Keys: category
- category MUST be exactly one of the allowed categories (exact string match).
- If you are unsure, choose the closest match (prefer 'Allgemein' if nothing fits well).

Return JSON:
{{"category":"..."}}
""".strip()

    resp = client.chat.completions.create(
        model=GWDG_MODEL,
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    txt = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if not m:
            return {"category": FALLBACK_CATEGORY}
        data = json.loads(m.group(0))

    cat = str(data.get("category", "")).strip()
    return {"category": cat or FALLBACK_CATEGORY}


def build_topic_category_map(
    topic_model: BERTopic,
    categories: List[str],
    cat_by_norm: Dict[str, str],
    gold_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    cache_path: Path,
) -> Dict[str, Dict[str, str]]:
    if FALLBACK_CATEGORY not in categories:
        raise ValueError(
            f"Fallback category '{FALLBACK_CATEGORY}' not found in your category list. "
            "Either add it to cropping_category.csv or change FALLBACK_CATEGORY."
        )

    cache = load_cache(cache_path)

    all_tids = set(
        collect_topic_ids_from_bertopic_topk(gold_df, TOP_K)
        + collect_topic_ids_from_bertopic_topk(unseen_df, TOP_K)
    )
    print(f"🔎 Unique BERTopic topic_ids to map: {len(all_tids)}")

    for tid in sorted(all_tids):
        key = str(int(tid))
        if key in cache:
            continue

        kw = topic_keywords_str(topic_model, tid, topn=12)
        mapped = gwdg_map_topic_to_category(tid, kw, categories)
        raw_cat = mapped["category"]

        # exact match first
        if raw_cat in categories:
            canonical = raw_cat
        else:
            # normalized match (handles umlauts/case/spacing)
            canonical = cat_by_norm.get(norm_key(raw_cat), "")

        # final fallback
        if not canonical:
            canonical = FALLBACK_CATEGORY
            print(f" Topic {tid}: model returned '{raw_cat}' -> fallback to '{FALLBACK_CATEGORY}'")

        cache[key] = {"category": canonical}
        save_cache(cache_path, cache)
        print(f" topic {tid} -> {canonical}")

    print(f"Topic→Category cache saved at: {cache_path}")
    return cache


# ===================== APPLY MAP =====================
def apply_category_map(df: pd.DataFrame, topic_map: Dict[str, Dict[str, str]], top_k: int) -> pd.DataFrame:
    df = df.copy()

    for i in range(1, top_k + 1):
        tid_col = f"bertopic_top{i}_topic_id"
        out_col = f"bertopic_top{i}_category"
        df[out_col] = df[tid_col].astype(int).astype(str).map(
            lambda k: topic_map.get(k, {}).get("category", FALLBACK_CATEGORY)
        )

    cat_cols = [f"bertopic_top{i}_category" for i in range(1, top_k + 1)]
    df["bertopic_topk_categories_joined"] = df[cat_cols].agg(", ".join, axis=1)

    score_cols = [f"bertopic_top{i}_topic_score" for i in range(1, top_k + 1)]
    final_ranked = []
    for cats, scores in zip(df[cat_cols].values, df[score_cols].values):
        agg: Dict[str, float] = {}
        for c, s in zip(cats, scores):
            agg[c] = agg.get(c, 0.0) + float(s)
        best = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_ranked.append([c for c, _ in best])

    for j in range(top_k):
        df[f"bertopic_final_cat_{j+1}"] = [row[j] if len(row) > j else "" for row in final_ranked]

    df["bertopic_final_categories_joined"] = df[[f"bertopic_final_cat_{j+1}" for j in range(top_k)]].agg(
        lambda r: ", ".join([x for x in r if x]), axis=1
    )

    return df


# ===================== MAIN =====================
def main():
    categories = load_categories(CATEGORY_CSV, CATEGORY_COL)
    cat_by_norm = {norm_key(c): c for c in categories}

    print("Loading BERTopic model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    topic_model = BERTopic.load(MODEL_DIR, embedding_model=embedding_model)
    print(" BERTopic model loaded.")

    gold_df = pd.read_csv(GOLD_CSV)
    unseen_df = pd.read_csv(UNSEEN_CSV)

    gold_df = add_bertopic_topk(gold_df, topic_model, GOLD_TEXT_COL, TOP_K)
    unseen_df = add_bertopic_topk(unseen_df, topic_model, UNSEEN_TEXT_COL, TOP_K)

    topic_map = build_topic_category_map(
        topic_model=topic_model,
        categories=categories,
        cat_by_norm=cat_by_norm,
        gold_df=gold_df,
        unseen_df=unseen_df,
        cache_path=TOPIC_MAP_CACHE,
    )

    gold_out = apply_category_map(gold_df, topic_map, TOP_K)
    unseen_out = apply_category_map(unseen_df, topic_map, TOP_K)

    gold_path = OUTPUT_DIR / "gold_bertopic_fixed_categories_topk5.csv"
    unseen_path = OUTPUT_DIR / "unseen_bertopic_fixed_categories_topk5.csv"
    gold_out.to_csv(gold_path, index=False)
    unseen_out.to_csv(unseen_path, index=False)

    print(f"Saved: {gold_path}")
    print(f"Saved: {unseen_path}")


if __name__ == "__main__":
    main()
