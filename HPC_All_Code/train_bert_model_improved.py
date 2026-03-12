#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# -------------------------
# Utils
# -------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        # Multi-label "accuracy" variants:
        "accuracy_subset": accuracy_score(y_true, y_pred),           # exact match of full labelset (strict)
        "accuracy_label_mean": float((y_true == y_pred).mean()),     # average per-label correctness
    }

def tune_global_threshold_for_f1_micro(y_true: np.ndarray, probs: np.ndarray, steps: int = 81):
    thresholds = np.linspace(0.10, 0.90, steps)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)

def tune_per_label_thresholds(y_true: np.ndarray, probs: np.ndarray, steps: int = 81):
    """
    For each label j, find threshold maximizing F1 for that label.
    """
    thresholds = np.linspace(0.10, 0.90, steps)
    L = y_true.shape[1]
    best = np.zeros(L, dtype=np.float32)
    best_f1s = np.zeros(L, dtype=np.float32)

    for j in range(L):
        yj = y_true[:, j]
        pj = probs[:, j]
        best_t, best_f1 = 0.5, -1.0
        for t in thresholds:
            pred = (pj >= t).astype(int)
            f1 = f1_score(yj, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best[j] = best_t
        best_f1s[j] = best_f1

    return best, best_f1s

def predict_with_per_label_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # probs [N,L], thresholds [L]
    return (probs >= thresholds.reshape(1, -1)).astype(int)

def predict_top_k(probs: np.ndarray, k: int) -> np.ndarray:
    # For each row, set top-k labels to 1
    N, L = probs.shape
    out = np.zeros((N, L), dtype=int)
    k = max(1, min(k, L))
    topk_idx = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
    for i in range(N):
        out[i, topk_idx[i]] = 1
    return out

def labels_to_string(label_names, y_row):
    return ", ".join([label_names[i] for i, v in enumerate(y_row) if v == 1])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="articles_with_categories.csv")
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--categories_col", type=str, default="categories")

    p.add_argument("--model_name", type=str, default="bert-base-german-cased")
    p.add_argument("--output_dir", type=str, default="./bert_multilabel_out")

    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Threshold tuning
    p.add_argument("--threshold_grid_steps", type=int, default=81)  # 0.10..0.90

    # Data fixes
    p.add_argument("--drop_allgemein_unless_only", action="store_true",
                   help="If set: remove 'Allgemein' from labels unless it is the only label.")
    p.add_argument("--min_label_count", type=int, default=1,
                   help="Drop labels that appear fewer than this many times in the FULL dataset. (Use 1 to keep all.)")

    # Decoding options for test outputs
    p.add_argument("--save_topk_predictions", action="store_true",
                   help="Also save a top-k decoding for test predictions.")
    p.add_argument("--top_k", type=int, default=3)

    # Iterative stratified split sizes
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.1)  # fraction of train (after test split)

    return p.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv_path)
    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df["input_text"] = df[args.text_col].str.strip()

    df[args.categories_col] = df[args.categories_col].fillna("").astype(str)
    df["labels_list"] = df[args.categories_col].apply(
        lambda s: [c.strip() for c in s.split(",") if c.strip()]
    )

    # Drop empty text or empty labels
    df = df[(df["input_text"].str.len() > 0) & (df["labels_list"].map(len) > 0)].reset_index(drop=True)

    # Optional: remove Allgemein unless only label
    if args.drop_allgemein_unless_only:
        def drop_allgemein(labs):
            if len(labs) <= 1:
                return labs
            return [l for l in labs if l != "Allgemein"]
        df["labels_list"] = df["labels_list"].apply(drop_allgemein)
        df = df[df["labels_list"].map(len) > 0].reset_index(drop=True)

    # Optional: drop extremely rare labels (helps stability)
    if args.min_label_count > 1:
        cnt = Counter([l for labs in df["labels_list"] for l in labs])
        keep = {l for l, c in cnt.items() if c >= args.min_label_count}
        df["labels_list"] = df["labels_list"].apply(lambda labs: [l for l in labs if l in keep])
        df = df[df["labels_list"].map(len) > 0].reset_index(drop=True)

    # Binarize
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels_list"]).astype(np.float32)
    label_names = list(mlb.classes_)
    num_labels = len(label_names)
    df["labels"] = list(Y)

    # Save label names
    with open(os.path.join(args.output_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)

    # ---------
    # Iterative stratified split (critical improvement)
    # ---------
    split_report = {}
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed
        )
        train_idx, test_idx = next(msss.split(df["input_text"], Y))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        Y_train = np.vstack(train_df["labels"].values)
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=args.val_size, random_state=args.seed
        )
        train_idx2, val_idx = next(msss2.split(train_df["input_text"], Y_train))
        val_df   = train_df.iloc[val_idx].reset_index(drop=True)
        train_df = train_df.iloc[train_idx2].reset_index(drop=True)

        split_report["split_method"] = "iterative_stratified"
    except Exception as e:
        # Fallback (worse): random split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, shuffle=True)
        train_df, val_df  = train_test_split(train_df, test_size=args.val_size, random_state=args.seed, shuffle=True)
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        split_report["split_method"] = "random_fallback"
        split_report["split_error"] = str(e)

    # Label coverage report (very important for diagnosing "can't predict label")
    def label_set(d):
        s = set()
        for labs in d["labels_list"]:
            s.update(labs)
        return s

    trainL, valL, testL = label_set(train_df), label_set(val_df), label_set(test_df)
    split_report.update({
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "num_labels_total": int(len(label_names)),
        "labels_in_train": int(len(trainL)),
        "labels_in_val": int(len(valL)),
        "labels_in_test": int(len(testL)),
        "labels_only_in_test": sorted(list(testL - trainL)),
        "labels_only_in_val": sorted(list(valL - trainL)),
    })

    with open(os.path.join(args.output_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(split_report, f, ensure_ascii=False, indent=2)

    # Convert to HF datasets
    train_ds = Dataset.from_pandas(train_df[["input_text", "labels"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df[["input_text", "labels"]], preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df[["input_text", "labels"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["input_text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["input_text"])
    val_ds   = val_ds.map(tokenize, batched=True).remove_columns(["input_text"])
    test_ds  = test_ds.map(tokenize, batched=True).remove_columns(["input_text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    # TrainingArguments compatibility: evaluation_strategy vs eval_strategy
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

    # ---------
    # Threshold tuning on validation
    # ---------
    val_out = trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_true = val_out.label_ids.astype(int)
    val_probs = sigmoid(val_logits)

    best_global_t, best_val_f1_micro = tune_global_threshold_for_f1_micro(
        y_true=val_true, probs=val_probs, steps=args.threshold_grid_steps
    )

    per_label_thresholds, per_label_f1s = tune_per_label_thresholds(
        y_true=val_true, probs=val_probs, steps=args.threshold_grid_steps
    )

    # Save thresholds
    with open(os.path.join(args.output_dir, "best_threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"{best_global_t}\n")

    per_label_thresholds_dict = {label_names[i]: float(per_label_thresholds[i]) for i in range(num_labels)}
    with open(os.path.join(args.output_dir, "per_label_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(per_label_thresholds_dict, f, ensure_ascii=False, indent=2)

    per_label_f1_dict = {label_names[i]: float(per_label_f1s[i]) for i in range(num_labels)}
    with open(os.path.join(args.output_dir, "val_per_label_f1_at_best_threshold.json"), "w", encoding="utf-8") as f:
        json.dump(per_label_f1_dict, f, ensure_ascii=False, indent=2)

    # Metrics on val
    val_pred_global = (val_probs >= best_global_t).astype(int)
    val_metrics_global = compute_all_metrics(val_true, val_pred_global)
    val_metrics_global.update({
        "best_global_threshold": best_global_t,
        "best_val_f1_micro_at_global_threshold": best_val_f1_micro,
    })

    val_pred_perlabel = predict_with_per_label_thresholds(val_probs, per_label_thresholds)
    val_metrics_perlabel = compute_all_metrics(val_true, val_pred_perlabel)
    val_metrics_perlabel.update({
        "thresholding": "per_label",
    })

    # ---------
    # Evaluate on test
    # ---------
    test_out = trainer.predict(test_ds)
    test_logits = test_out.predictions
    test_true = test_out.label_ids.astype(int)
    test_probs = sigmoid(test_logits)

    test_pred_global = (test_probs >= best_global_t).astype(int)
    test_metrics_global = compute_all_metrics(test_true, test_pred_global)
    test_metrics_global.update({"used_global_threshold": best_global_t})

    test_pred_perlabel = predict_with_per_label_thresholds(test_probs, per_label_thresholds)
    test_metrics_perlabel = compute_all_metrics(test_true, test_pred_perlabel)
    test_metrics_perlabel.update({"thresholding": "per_label"})

    metrics = {
        "num_labels": num_labels,
        "labels": label_names,
        "split_report": split_report,
        "val": {
            "global_threshold": val_metrics_global,
            "per_label_thresholds": val_metrics_perlabel,
        },
        "test": {
            "global_threshold": test_metrics_global,
            "per_label_thresholds": test_metrics_perlabel,
        },
        "train_args": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_bs": args.train_bs,
            "eval_bs": args.eval_bs,
            "seed": args.seed,
            "drop_allgemein_unless_only": bool(args.drop_allgemein_unless_only),
            "min_label_count": int(args.min_label_count),
        },
    }

    # top-k decoding metrics for test (useful for visualization)
    if args.save_topk_predictions:
        test_pred_topk = predict_top_k(test_probs, args.top_k)
        test_metrics_topk = compute_all_metrics(test_true, test_pred_topk)
        metrics["test"]["top_k"] = {"k": int(args.top_k), **test_metrics_topk}

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save test predictions (multiple decoding variants)
    pred_df = pd.DataFrame({
        "text": test_df["input_text"].values,
        "true_labels": test_df["labels_list"].apply(lambda x: ", ".join(x)).values,
        "pred_labels_global_threshold": [labels_to_string(label_names, row) for row in test_pred_global],
        "pred_labels_per_label_thresholds": [labels_to_string(label_names, row) for row in test_pred_perlabel],
    })

    if args.save_topk_predictions:
        test_pred_topk = predict_top_k(test_probs, args.top_k)
        pred_df[f"pred_labels_topk_{args.top_k}"] = [labels_to_string(label_names, row) for row in test_pred_topk]

    pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False, encoding="utf-8")

    print("Done.")
    print("Saved everything to:", args.output_dir)
    print("Split method:", split_report.get("split_method"))
    print("Labels only in test:", len(split_report.get("labels_only_in_test", [])))
    print("Best global threshold:", best_global_t, "val f1_micro:", best_val_f1_micro)
    print("VAL (global):", val_metrics_global)
    print("VAL (per-label):", val_metrics_perlabel)
    print("TEST (global):", test_metrics_global)
    print("TEST (per-label):", test_metrics_perlabel)

if __name__ == "__main__":
    main()
