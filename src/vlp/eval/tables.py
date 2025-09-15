from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

def to_markdown_table(df: pd.DataFrame, dataset_key: str) -> str:
    # Pivot so methods are columns
    p = df[df["dataset"]==dataset_key].pivot(index=["fraction","nodes"], columns="method", values="time_s").sort_index()
    # Replace NaN with "X" for failures
    p = p.fillna("X")
    # Format numbers
    def fmt(x):
        if isinstance(x, str):
            return x
        try:
            return f"{x:.2f}"
        except Exception:
            return x
    p = p.applymap(fmt)
    return p.to_markdown()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="results CSV from runner")
    ap.add_argument("--dataset", type=str, required=True, help="dataset key (amazon_products|yelp|taobao|flickr)")
    ap.add_argument("--out", type=str, default=None, help="optional markdown path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    md = to_markdown_table(df, args.dataset)
    if args.out:
        Path(args.out).write_text(md)
    else:
        print(md)

if __name__ == "__main__":
    main()
