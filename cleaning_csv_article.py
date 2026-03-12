# -*- coding: utf-8 -*-
import re
import pandas as pd
from pathlib import Path

# ====== CONFIG ======
INPUT_DIR  = Path("./label_article/newspaper_articles/output_articles_csv")
OUTPUT_DIR = Path("./label_article/newspaper_articles/output_articles_csv_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Cleaning Utilities ----------

def collapse_repeated_punct(text: str) -> str:
    """Collapse runs of the same punctuation into a single character."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'([!?.:,;()\[\]\"\'\-])\1+', r'\1', text)


def remove_boilerplate_and_page_nums(text: str) -> str:
    """Remove Fortsetzung lines, page numbers, and digit-only lines."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove '.. Fortsetzung...' or 'Fortsetzung...' tails (case-insensitive)
    text = re.sub(r'(?im)^.*fortsetzung.*$', '', text)

    # remove "Seite <num> von <num>"
    text = re.sub(r'(?i)\bseite\s*\d+\s*von\s*\d+\b', '', text)

    # remove simple "Seite <num>"
    text = re.sub(r'(?i)\bseite\s*\d+\b', '', text)

    # remove lines that are ONLY digits
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)

    return text


def remove_until_after_auflage(text: str) -> str:
    """
    Remove everything before the first occurrence of 'Auflage',
    including the 'Auflage' line and the immediately following line/sentence.
    Example:
        Universität Potsdam
        Quelle: swr.de vom 17.06.2020
        Auflage:
        SWR»
        Im Flow - Mit Selbstvergessenheit zum Erfolg?
    ➜ becomes:
        Im Flow - Mit Selbstvergessenheit zum Erfolg?
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned = re.sub(
        r"(?is)^.*?auflage[^\n]*\n[^\n]*\n?",  # remove before & including next line
        "",
        text,
        count=1,
    )

    return cleaned.strip()


def fix_hyphenation(text: str) -> str:
    """Join words split across lines with hyphen + space/newline."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)


def normalize_unicode(text: str) -> str:
    """Normalize special quotes/dashes to standard ASCII where possible."""
    if not isinstance(text, str):
        return ""
    text = text.replace("„", '"').replace("“", '"').replace("‚", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces and blank lines."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00A0", " ")
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()


# ---------- Main Cleaning Pipeline ----------

def clean_article(text: str) -> str:
    """Full cleaning pipeline for one article."""
    if not isinstance(text, str):
        return ""
    t = collapse_repeated_punct(text)
    t = remove_boilerplate_and_page_nums(t)
    t = remove_until_after_auflage(t)       
    t = fix_hyphenation(t)
    # fix joined words like 'WaldbesitzerninBrandenburg'
    t = re.sub(r'([a-zäöüß])([A-ZÄÖÜ])', r'\1 \2', t)
    # merge broken sentences split by accidental line breaks
    t = re.sub(r'([a-zäöüß])\n([a-zäöüß])', r'\1 \2', t)
    t = normalize_unicode(t)
    t = normalize_whitespace(t)
    return t.strip()


# ---------- Batch Processing ----------

def process_csv(in_csv: Path, out_csv: Path):
    df = pd.read_csv(in_csv)

    # drop the first row (if it's not an article)
    if len(df) > 0:
        df = df.iloc[1:].copy()

    if "article" not in df.columns:
        raise ValueError(f"{in_csv.name}: 'article' column not found. Columns: {df.columns.tolist()}")

    df["clean_article"] = df["article"].apply(clean_article)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] {in_csv.name} -> {out_csv.name}  ({len(df)} rows)")


# ---------- Run over all CSVs ----------

csv_files = sorted(INPUT_DIR.glob("*.csv"))
if not csv_files:
    print(f"No CSVs found in {INPUT_DIR}")
else:
    for in_path in csv_files:
        out_path = OUTPUT_DIR / in_path.name
        try:
            process_csv(in_path, out_path)
        except Exception as e:
            print(f"[ERROR] {in_path.name}: {e}")

print(" Done.")
