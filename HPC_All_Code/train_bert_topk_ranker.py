#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--id_col", type=str, default="id")              # file uses `id`
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="category")     # one label per row
    p.add_argument("--model_name", type=str, default="bert-base-german-cased")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=5)
    return p.parse_args()

def softmax(x: np.ndarray, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)

def topk_metrics(true_sets, pred_lists, k: int):
    # true_sets: list[set[str]], pred_lists: list[list[str]]
    precs, recs, f1s, exact = [], [], [], []
    for tset, preds in zip(true_sets, pred_lists):
        preds = preds[:k]
        pset = set(preds)
        tp = len(pset & tset)
        prec = tp / max(len(pset), 1)
        rec  = tp / max(len(tset), 1)
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
        exact.append(1.0 if pset == tset else 0.0)
    return {
        f"precision@{k}": float(np.mean(precs)),
        f"recall@{k}": float(np.mean(recs)),
        f"f1@{k}": float(np.mean(f1s)),
        "subset_accuracy": float(np.mean(exact)),
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df[args.label_col] = df[args.label_col].fillna("").astype(str).str.strip()
    df = df[(df[args.text_col].str.len() > 0) & (df[args.label_col].str.len() > 0)].reset_index(drop=True)

    # Build label vocab
    label_names = sorted(df[args.label_col].unique().tolist())
    label2id = {l:i for i,l in enumerate(label_names)}
    id2label = {i:l for l,i in label2id.items()}

    with open(os.path.join(args.output_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)

    # === split by article id to avoid leakage ===
    unique_ids = df[args.id_col].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=args.seed, shuffle=True)
    train_ids, val_ids  = train_test_split(train_ids, test_size=0.1, random_state=args.seed, shuffle=True)

    train_df = df[df[args.id_col].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[args.id_col].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df[args.id_col].isin(test_ids)].reset_index(drop=True)

    # Save split report
    split_report = {
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_val": int(len(val_df)),
        "n_rows_test": int(len(test_df)),
        "n_articles_total": int(len(unique_ids)),
        "n_articles_train": int(len(train_ids)),
        "n_articles_val": int(len(val_ids)),
        "n_articles_test": int(len(test_ids)),
        "num_labels": int(len(label_names)),
    }
    with open(os.path.join(args.output_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(split_report, f, ensure_ascii=False, indent=2)

    # Encode labels (single integer)
    def encode(df_):
        out = df_.copy()
        out["label_id"] = out[args.label_col].map(label2id).astype(int)
        return out

    train_df = encode(train_df)
    val_df   = encode(val_df)
    test_df  = encode(test_df)

    # HF datasets
    train_ds = Dataset.from_pandas(train_df[[args.id_col, args.text_col, "label_id"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df[[args.id_col, args.text_col, "label_id"]], preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df[[args.id_col, args.text_col, "label_id"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tok(batch):
        return tokenizer(batch[args.text_col], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tok, batched=True)
    val_ds   = val_ds.map(tok, batched=True)
    test_ds  = test_ds.map(tok, batched=True)

    # Rename to Trainer expected column
    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds   = val_ds.rename_column("label_id", "labels")
    test_ds  = test_ds.rename_column("label_id", "labels")

    # Keep id for grouping later
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    eval_kw = {"evaluation_strategy": "epoch"} if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames else {"eval_strategy": "epoch"}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        **eval_kw,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # ===== Article-level top-k evaluation on TEST =====
    # Build article->true label-set from exploded test_df
    true_by_article = test_df.groupby(args.id_col)[args.label_col].apply(lambda s: set(s.tolist())).to_dict()

    # Predict once per article: take first row text as representative
    # (all rows share same text for the article_id)
    rep = test_df.groupby(args.id_col).head(1)[[args.id_col, args.text_col]].reset_index(drop=True)
    rep_ds = Dataset.from_pandas(rep, preserve_index=False).map(
        lambda b: tokenizer(b[args.text_col], truncation=True, max_length=args.max_length),
        batched=True
    )

    out = trainer.predict(rep_ds)
    probs = softmax(out.predictions, axis=1)

    k = args.top_k
    topk_idx = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
    pred_lists = []
    for i in range(probs.shape[0]):
        ii = topk_idx[i]
        ii = ii[np.argsort(-probs[i, ii])]
        pred_lists.append([label_names[j] for j in ii])

    rep_ids = rep[args.id_col].tolist()
    true_sets = [true_by_article[i] for i in rep_ids]

    metrics = {"topk_test": topk_metrics(true_sets, pred_lists, k)}
    with open(os.path.join(args.output_dir, "metrics_topk.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save article-level test predictions
    pred_df = rep.copy()
    pred_df["true_labels"] = [", ".join(sorted(true_by_article[i])) for i in rep_ids]
    pred_df["pred_labels_topk"] = [", ".join(p) for p in pred_lists]
    pred_df["pred_label_count"] = k
    pred_df["used_threshold"] = f"top_k={k}"
    pred_df.to_csv(os.path.join(args.output_dir, "test_predictions_topk.csv"), index=False, encoding="utf-8")

    # Save model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done. Saved to:", args.output_dir)
    print("Top-k test metrics:", metrics["topk_test"])

if __name__ == "__main__":
    main()
