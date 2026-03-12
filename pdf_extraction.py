"""
pdf_extraction.py
----------------
Convert a PDF into OCR-structured JSON using Tesseract.

Features
- Rasterizes PDF pages into images (via pdf2image).
- Runs Tesseract OCR (block-level mode, psm 1) on each page.
- Groups detected words into text blocks.
- Exports a JSON file where each page contains its recognized blocks.

JSON Output Format
------------------
[
  {"page": 1, "blocks": ["block text 1", "block text 2", ...]},
  {"page": 2, "blocks": [...]},
  ...
]

Usage
-----
python pdf_to_blocks.py --pdf ./media/Pressespiegel_20200602.pdf \
                        --out ./label_article/sample_articles/output_articles/tesseract_blocks.json

Dependencies
------------
pip install pdf2image pytesseract pillow

System packages
---------------
- Tesseract OCR must be installed and available in PATH (or set TESSERACT_CMD env).
  On macOS: brew install tesseract
- Poppler is required by pdf2image for PDF rasterization.
  On macOS: brew install poppler
  On Linux: apt-get install poppler-utils

"""


from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import json
import glob
import time

PDF_FOLDER = "./media/Subset_newspaper"   # Folder containing all PDFs
OUTPUT_DIR = "./label_article/newspaper_articles/output_articles_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all PDF files
pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
total_files = len(pdf_files)
processed = 0
skipped = 0
failed = 0

start_time = time.time()

print(f"Found {total_files} PDF files to process.\n")

for index, pdf_path in enumerate(pdf_files, start=1):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{pdf_name}_tesseract_blocks.json")

    # Skip already processed files
    if os.path.exists(output_file):
        print(f"[{index}/{total_files}] Skipping already processed file: {pdf_name}")
        skipped += 1
        continue

    print(f"\n[{index}/{total_files}] Processing file: {pdf_name}")

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f" Skipping missing file: {pdf_path}")
        failed += 1
        continue

    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f" Skipping {pdf_name} due to PDF conversion error: {e}")
        failed += 1
        continue

    all_pages = []

    for page_num, image in enumerate(images):
        print(f" OCR on page {page_num + 1}")
        try:
            # OCR with Tesseract
            ocr_result = pytesseract.image_to_data(
                image,
                lang='deu',
                output_type=pytesseract.Output.DICT,
                config="--psm 1"
            )
        except Exception as e:
            print(f" Skipping page {page_num + 1} due to OCR error: {e}")
            continue

        blocks = []
        current_text = ""
        last_block_num = -1

        for i in range(len(ocr_result['text'])):
            word = ocr_result['text'][i]
            block_num = ocr_result['block_num'][i]

            if word.strip() == "":
                continue

            if block_num != last_block_num:
                if current_text:
                    blocks.append(current_text.strip())
                current_text = word
                last_block_num = block_num
            else:
                current_text += " " + word

        if current_text:
            blocks.append(current_text.strip())

        all_pages.append({
            "page": page_num + 1,
            "blocks": blocks
        })

    # Save results
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_pages, f, indent=2, ensure_ascii=False)
        processed += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / (processed + skipped + failed)
        remaining = avg_time * (total_files - (processed + skipped + failed))
        print(f"  Done! Saved to {output_file}")
        print(f"  Progress: {processed}/{total_files} processed | {skipped} skipped | {failed} failed | ETA: {remaining/60:.1f} min")
    except Exception as e:
        print(f"  Failed to save JSON for {pdf_name}: {e}")
        failed += 1

# Final summary
end_time = time.time()
duration = (end_time - start_time) / 60

print("\n All PDFs processed!")
print(f" Summary: {processed} processed | {skipped} skipped | {failed} failed | Total time: {duration:.1f} min")
