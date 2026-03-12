#!/usr/bin/env python3
"""
Parallel & checkpointed LLM labeling for German newspaper articles using GWDG API
MULTI-LABEL classification with a fixed category list (from cropping_category.csv)
Output format: comma-separated labels, e.g.
Allgemein, Bildungswissenschaften, Forschung, Humanwissenschaftliche Fakultät
"""

import os
import time
import random
import re
import pandas as pd
from pathlib import Path
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# CONFIGURATION
# ==============================
openai.api_base = "GWDG_Link"
openai.api_key = os.getenv("GWDG_API_KEY", "key")

INPUT_DIR = Path("./label_article/newspaper_articles/output_articles_csv_clean")
OUTPUT_DIR = Path("./label_article/final_label/gwdg_unseen_articles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Category file 
CATEGORY_FILE = Path("./label_article/newspaper_articles/cropping_category.csv")


MODEL = "llama-3.3-70b-instruct"
MAX_WORKERS = 4
BASE_DELAY = 10
RETRIES = 5
CHECKPOINT_INTERVAL = 100

# EXACT top_k labels per article
TOP_K = 5

# Article text column name
ARTICLE_COLUMN = "clean_article"

# ==============================
# LOAD CATEGORIES
# ==============================
_cat_df = pd.read_csv(CATEGORY_FILE)
if "name" not in _cat_df.columns:
    raise ValueError(f"Expected a 'name' column in {CATEGORY_FILE}, found: {_cat_df.columns.tolist()}")

CATEGORIES = (
    _cat_df["name"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

if len(CATEGORIES) < TOP_K:
    raise ValueError(f"Category list has only {len(CATEGORIES)} categories, but TOP_K={TOP_K} requested.")

CATEGORY_SET = set(CATEGORIES)
CATEGORY_LIST_TEXT = "\n".join(f"- {c}" for c in CATEGORIES)

# Fallback label used only if the model returns fewer than TOP_K valid labels
FALLBACK_LABEL = "Allgemein" if "Allgemein" in CATEGORY_SET else CATEGORIES[0]

# ==============================
# HELPERS
# ==============================
def _parse_labels(raw: str):
    """
    Parse model output into a list of valid category names.
    Expected model format: "Label1, Label2, Label3" (comma-separated).
    Returns only labels that match CATEGORY_SET exactly.
    """
    raw = (raw or "").strip()

    if raw == "" or raw == "[]":
        return []

    # If model accidentally outputs a JSON-like list, normalize
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw.strip("[]").strip()
        parts = re.split(r"\s*,\s*", inner)
        parts = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
        labels = parts
    else:
        labels = [p.strip() for p in re.split(r"\s*,\s*", raw) if p.strip()]

    # Keep only exact matches + de-duplicate preserving order
    out = []
    seen = set()
    for l in labels:
        if l in CATEGORY_SET and l not in seen:
            out.append(l)
            seen.add(l)
    return out


def _ensure_exact_top_k(labels):
    """Enforce EXACTLY TOP_K labels (unique). Pad deterministically if too few."""
    labels = list(dict.fromkeys(labels))  # keep order, drop duplicates
    labels = labels[:TOP_K]

    if len(labels) < TOP_K:
        # First, try fallback label
        if FALLBACK_LABEL not in labels:
            labels.append(FALLBACK_LABEL)

        # Then add other categories deterministically (first in list not already used)
        if len(labels) < TOP_K:
            for c in CATEGORIES:
                if c not in labels:
                    labels.append(c)
                if len(labels) == TOP_K:
                    break

    return labels[:TOP_K]


def label_article(text, model=MODEL, retries=RETRIES):
    """Send one article to the GWDG LLM API and return EXACTLY TOP_K categories (ordered)."""
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You assign topic categories to German newspaper articles.\n"
                            "This is MULTI-LABEL classification.\n"
                            "You MUST choose ONLY from the provided category list.\n"
                            f"Return EXACTLY {TOP_K} categories, ordered from most relevant to least relevant.\n"
                            f"Even if fewer seem relevant, you MUST still output {TOP_K} best-matching categories.\n"
                            "Return ONLY category names separated by comma+space.\n"
                            "Do NOT add any extra words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Category list:\n{CATEGORY_LIST_TEXT}\n\n"
                            "Task:\n"
                            f"- Return exactly {TOP_K} categories.\n"
                            "- Use exact category names from the list.\n"
                            f"- Output format: Label1, Label2, Label3, Label4, Label5 (comma+space).\n"
                            "- Order from most relevant to least relevant.\n\n"
                            f"Article:\n{text}"
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=200,
            )

            raw = response["choices"][0]["message"]["content"].strip()
            labels = _parse_labels(raw)
            labels = _ensure_exact_top_k(labels)

            # Throttle requests
            time.sleep(BASE_DELAY + random.uniform(0, 2))
            return labels

        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower() or "429" in msg:
                wait = 180 * (attempt + 1) + random.randint(0, 5)
                print(f"[WARN] Rate limit: waiting {wait}s before retry {attempt+1}")
                time.sleep(wait)
            else:
                print(f"[WARN] API error on attempt {attempt+1}: {e}")
                time.sleep(2 * (attempt + 1))

    return ["[ERROR]"]


# ==============================
# MAIN PROCESSING FUNCTION
# ==============================
def process_file(csv_file: Path):
    """Process one CSV file of articles and handle [ERROR] relabeling safely."""
    out_path = OUTPUT_DIR / csv_file.name

    # ==============================
    # SMART SKIP / REPROCESS LOGIC
    # ==============================
    if out_path.exists():
        try:
            df_existing = pd.read_csv(out_path)
        except Exception as e:
            print(f"[ERROR] Could not read existing output {out_path}: {e}")
            return

        if "llm_labels" not in df_existing.columns:
            print(f"[REPROCESS] {csv_file.name} output exists but missing llm_labels — reprocessing.")
            df = df_existing.copy()
            df["llm_labels"] = ""
        else:
            if not df_existing["llm_labels"].astype(str).str.contains(r"\[ERROR\]", na=False).any():
                print(f"[SKIP] {csv_file.name} already processed (no errors).")
                return
            else:
                print(f"[REPROCESS] {csv_file.name} contains [ERROR] labels — retrying those rows.")
                df = df_existing.copy()
    else:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[ERROR] Could not read {csv_file}: {e}")
            return

    # ==============================
    # PREPARE DATAFRAME
    # ==============================
    if ARTICLE_COLUMN not in df.columns:
        print(f"[ERROR] {csv_file.name} missing required column '{ARTICLE_COLUMN}'")
        return

    if "llm_labels" not in df.columns:
        df["llm_labels"] = ""

    # Identify remaining or failed rows
    remaining = df[
        df["llm_labels"].isna()
        | (df["llm_labels"].astype(str).str.strip() == "")
        | (df["llm_labels"].astype(str).str.contains(r"\[ERROR\]", na=False))
    ]

    total = len(df)
    remaining_count = len(remaining)

    if remaining_count == 0:
        print(f"[DONE] All {total} articles already labeled in {csv_file.name}")
        return

    print(f"→ {csv_file.name}: {remaining_count}/{total} articles to process")

    # ==============================
    # PARALLEL LABELING
    # ==============================
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(label_article, text): idx
            for idx, text in remaining[ARTICLE_COLUMN].items()
        }
        completed = 0

        for future in as_completed(futures):
            idx = futures[future]
            labels = future.result()

            if labels == ["[ERROR]"]:
                df.at[idx, "llm_labels"] = "[ERROR]"
            else:
                df.at[idx, "llm_labels"] = ", ".join(labels)  # always exactly TOP_K

            completed += 1

            if completed % 20 == 0:
                print(f"Progress: {completed}/{remaining_count} labeled")

            # Checkpoint save
            if completed % CHECKPOINT_INTERVAL == 0:
                temp_path = out_path.with_suffix(".partial.csv")
                df.to_csv(temp_path, index=False)
                print(f"[CHECKPOINT] Saved intermediate results → {temp_path}")

    # ==============================
    # SAVE FINAL OUTPUT
    # ==============================
    df.to_csv(out_path, index=False)
    print(f"[OK] Finished {csv_file.name} → {out_path}")


# ==============================
# ENTRY POINT
# ==============================
def main():
    if not openai.api_key:
        print("[ERROR] Missing GWDG_API_KEY env var. Example:")
        print("  export GWDG_API_KEY='your_key_here'")
        return

    csv_files = list(INPUT_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {INPUT_DIR}")

    for csv_file in csv_files:
        process_file(csv_file)


if __name__ == "__main__":
    main()