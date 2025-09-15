#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["flickr","amazon","yelp","taobao"])
    ap.add_argument("--root", type=str, default="data")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if "flickr" in args.datasets:
        from vlp.data.flickr import load_flickr
        try:
            X, meta = load_flickr(str(root))
            print(f"[flickr] features: {tuple(X.shape)}")
        except Exception as e:
            print(f"[flickr] error: {e}")

    if "amazon" in args.datasets:
        from vlp.data.amazon_products import load_amazon_products
        try:
            X, meta = load_amazon_products(str(root))
            print(f"[amazon] features: {tuple(X.shape)}")
        except Exception as e:
            print(f"[amazon] error: {e}")

    if "yelp" in args.datasets:
        from vlp.data.yelp import load_yelp
        try:
            X, meta = load_yelp(str(root))
            print(f"[yelp] features: {tuple(X.shape)}")
        except Exception as e:
            print(f"[yelp] error: {e}")

    if "taobao" in args.datasets:
        base = root / "taobao"
        base.mkdir(parents=True, exist_ok=True)
        print("[taobao] Place required CSVs under data/taobao/: user_item.csv and item_category.csv (see paper reference).")
        print("[taobao] Once present, the loader will build a dense TF-IDF user×category matrix locally.")

if __name__ == "__main__":
    main()
