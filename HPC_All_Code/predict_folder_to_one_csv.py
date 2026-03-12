#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="Fine-tuned model directory")
    p.add_argument("--input_dir", type=str, required=True, help="Folder containing many CSV files")
    p.add_argument("--output_csv", type=str, required=True, help="Single merged output CSV")
    p.add_argument("--text_col", type=str, default="clean_article", help="Column to use as input text")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--glob_pattern", type=str, default="*.csv", help="Pattern for files in input_dir")
    p.add_argument("--add_source_file", action="store_true",
                   help="Add a column source_file with the filename each row came from")
    return p.parse_args()

@torch.no_grad()
def predict_texts(model, tokenizer, texts, device, max_length, batch_size):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        all_probs.append(sigmoid(logits))
    return np.vstack(all_probs)

def main():
    args = parse_args()

    label_path = os.path.join(args.model_dir, "label_names.json")
    thr_path = os.path.join(args.model_dir, "best_threshold.txt")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing {label_path}")
    if not os.path.exists(thr_path):
        raise FileNotFoundError(f"Missing {thr_path}")

    label_names = json.load(open(label_path, "r", encoding="utf-8"))
    threshold = float(open(thr_path, "r", encoding="utf-8").read().strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    model.to(device)

    pattern = os.path.join(args.input_dir, args.glob_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found with pattern: {pattern}")

    out_parts = []
    skipped = 0

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[SKIP] Could not read {fp}: {e}")
            skipped += 1
            continue

        if args.text_col not in df.columns:
            print(f"[SKIP] {fp}: missing column '{args.text_col}' (has {list(df.columns)[:10]}...)")
            skipped += 1
            continue

        texts = df[args.text_col].fillna("").astype(str).tolist()

        probs = predict_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        pred = (probs >= threshold).astype(int)

        pred_labels = []
        pred_counts = []
        for row in pred:
            labs = [label_names[j] for j, v in enumerate(row) if v == 1]
            pred_labels.append(", ".join(labs))
            pred_counts.append(len(labs))

        df_out = df.copy()
        df_out["pred_labels"] = pred_labels
        df_out["pred_label_count"] = pred_counts
        df_out["used_threshold"] = threshold

        if args.add_source_file:
            df_out["source_file"] = os.path.basename(fp)

        out_parts.append(df_out)

        print(f"[OK] {os.path.basename(fp)} rows={len(df_out)} device={device}")

    if not out_parts:
        raise RuntimeError("No usable CSVs were processed (all skipped).")

    merged = pd.concat(out_parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    merged.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Processed files: {len(out_parts)} / {len(files)} (skipped {skipped})")
    print(f"Total rows: {len(merged)}")
    print(f"Saved: {args.output_csv}")
    print(f"Threshold: {threshold} | Labels: {len(label_names)} | Device: {device}")

if __name__ == "__main__":
    main()
