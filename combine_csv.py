#!/usr/bin/env python3
"""
Combine multiple CSV files from a fixed input directory into a single CSV
written to a fixed output directory.

Run:
  python combine_csv.py
"""

import os
import glob
import pandas as pd


# =========================
# CONFIG: set your paths here
# =========================
INPUT_DIR = "./label_article/final_label/gwdg_unseen_articles"
OUTPUT_DIR = "./label_article/final_label/combine_csv"
OUTPUT_NAME = "gwdg_combine_real_article.csv"
# =========================


def combine_csvs(input_dir: str, output_dir: str, output_name: str) -> str:
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["__source_file"] = os.path.basename(path)  
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=True)

    out_path = os.path.join(output_dir, output_name)
    combined.to_csv(out_path, index=False)
    return out_path


def main():
    out_path = combine_csvs(INPUT_DIR, OUTPUT_DIR, OUTPUT_NAME)
    print(f" Combined CSV saved to: {out_path}")


if __name__ == "__main__":
    main()
