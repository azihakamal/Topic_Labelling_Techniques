import json
import pandas as pd
from pathlib import Path

def extract_articles(in_file, out_json=None, out_csv=None, tail_chars=250):
    """
    Extract articles from OCR-style JSON.
    Groups pages into articles using continuation and new-article heuristics.

    Rules:
      1) If a page starts with 'Fortsetzung' or the previous page ended with 'Fortsetzung',
         then this page continues the same article.
      2) If a page contains 'Quelle:' near the top OR looks like a headline (short, capitalized),
         it forces the start of a new article.
      3) Table-of-contents style pages (with 'Fortsetzung' in early blocks) merge correctly.

    Args:
        in_file (str or Path): input JSON path
        out_json (str or Path, optional): path to save JSON output
        out_csv (str or Path, optional): path to save CSV output
        tail_chars (int): trailing character window to detect 'fortsetzung' at page end

    Returns:
        list of dicts: [{"id": int, "page": "start-end", "article": str}, ...]
    """
    in_path = Path(in_file)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    def join_blocks(blocks):
        return "\n".join(b for b in blocks if isinstance(b, str) and b.strip())

    def has_tail_fortsetzung(txt: str, tail=tail_chars) -> bool:
        """Detect 'fortsetzung' near the very end of a page."""
        if not txt:
            return False
        last_line = txt.splitlines()[-1].lower()
        return "fortsetzung" in last_line or "fortsetzung" in txt[-tail:].lower()

    def is_new_article_start(blocks):
        """Heuristic: decide if a page starts a new article."""
        if not blocks:
            return False
        first_line = blocks[0].strip().lower()

        # If starts with 'fortsetzung', it's a continuation not a new start
        if "fortsetzung" in first_line:
            return False

        # "Quelle:" very early is a strong signal for new article
        for b in blocks[:3]:
            if "quelle:" in b.lower():
                return True

        # Title-like headline (short, capitalized, no colon at end)
        if len(blocks[0].split()) < 8 and blocks[0][0].isupper() and not blocks[0].endswith(":"):
            return True

        return False

    articles = []
    cur, next_id = None, 0
    force_continue_next = False

    for page in data:
        page_no = page.get("page")
        blocks = page.get("blocks", [])
        if not blocks:
            continue

        page_text = join_blocks(blocks)
        first_line = blocks[0].splitlines()[0].strip() if blocks and isinstance(blocks[0], str) else ""

        # --- Decide continuation vs. new start ---
        is_cont = force_continue_next or ("fortsetzung" in first_line.lower())

        # Override: force new article if this page clearly starts fresh
        if is_new_article_start(blocks):
            is_cont = False
            force_continue_next = False

        if cur is None or not is_cont:
            # Close the previous article
            if cur is not None and cur["text"].strip():
                start, end = min(cur["pages"]), max(cur["pages"])
                articles.append({
                    "id": cur["id"],
                    "page": f"{start}-{end}" if start != end else str(start),
                    "article": cur["text"].strip()
                })
            # Start a new article
            cur = {"id": next_id, "pages": set(), "text": ""}
            next_id += 1

        # Add this page to the current article
        cur["pages"].add(page_no)
        cur["text"] += ("" if not cur["text"] else "\n") + page_text

        # Set carry-over for next page
        force_continue_next = has_tail_fortsetzung(page_text)

    # Flush the last article
    if cur and cur["text"].strip():
        start, end = min(cur["pages"]), max(cur["pages"])
        articles.append({
            "id": cur["id"],
            "page": f"{start}-{end}" if start != end else str(start),
            "article": cur["text"].strip()
        })

    # Save outputs if requested
    if out_json:
        Path(out_json).write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(articles).to_csv(out_csv, index=False, encoding="utf-8")

    return articles

# ---------- Batch convert all JSON -> CSV (keep original names) ----------
INPUT_DIR = Path("./label_article/newspaper_articles/output_articles_json")
OUTPUT_DIR = Path("./label_article/newspaper_articles/output_articles_csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0
for in_path in sorted(INPUT_DIR.glob("*.json")):
    base = in_path.stem                  # original name without extension
    out_csv = OUTPUT_DIR / f"{base}.csv" # keep same name, change extension
    try:
        arts = extract_articles(in_path, out_csv=out_csv, tail_chars=250)  # adjust tail_chars if needed
        print(f"[OK] {in_path.name} -> {out_csv.name} ({len(arts)} articles)")
        count += 1
    except Exception as e:
        print(f"[ERROR] {in_path.name}: {e}")

print(f"Done. Converted {count} JSON file(s).")
