#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def softmax(x: np.ndarray, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--text_col", default="clean_article")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()

@torch.no_grad()
def predict_probs(model, tokenizer, texts, device, max_length, batch_size):
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        probs_all.append(softmax(logits, axis=1))
    return np.vstack(probs_all)

def main():
    args = parse_args()

    label_path = os.path.join(args.model_dir, "label_names.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing {label_path}")
    label_names = json.load(open(label_path, "r", encoding="utf-8"))

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    texts = df[args.text_col].fillna("").astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    probs = predict_probs(model, tokenizer, texts, device, args.max_length, args.batch_size)

    k = max(1, min(args.top_k, probs.shape[1]))
    topk_idx = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]

    pred_labels = []
    for i in range(probs.shape[0]):
        ii = topk_idx[i]
        ii = ii[np.argsort(-probs[i, ii])]
        pred_labels.append(", ".join(label_names[j] for j in ii))

    out = df.copy()
    out["pred_labels"] = pred_labels
    out["pred_label_count"] = k
    out["used_threshold"] = f"top_k={k}"

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out.to_csv(args.output_csv, index=False, encoding="utf-8")
    print("DONE")
    print("Input:", args.input_csv)
    print("Output:", args.output_csv)
    print("Rows:", len(out))

if __name__ == "__main__":
    main()
