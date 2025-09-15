#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))
from pathlib import Path
import pandas as pd
from vlp.eval.tables import to_markdown_table

def main():
    outdir = Path("outputs")
    for key in ["amazon", "yelp", "taobao"]:
        csv = outdir / f"{key}_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            md = to_markdown_table(df, key if key != "amazon" else "amazon_products")
            (outdir / f"{key}_table.md").write_text(md)
            print(f"[tables] wrote {outdir / f'{key}_table.md'}")

if __name__ == "__main__":
    main()
