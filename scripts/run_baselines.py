#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
import torch
import numpy as np

from vlp.baselines.sklearn_lp import run_sklearn_lp
from vlp.baselines.sknetwork_lp import run_sknetwork_lp
from vlp.baselines.pyg_lp import run_pyg_lp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="flickr", choices=["flickr","amazon","yelp","taobao"])
    ap.add_argument("--fraction", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.dataset == "flickr":
        from vlp.data.flickr import load_flickr as loader
    elif args.dataset == "amazon":
        from vlp.data.amazon_products import load_amazon_products as loader
    elif args.dataset == "yelp":
        from vlp.data.yelp import load_yelp as loader
    else:
        from vlp.data.taobao import load_taobao as loader

    X, meta = loader("data")
    n = X.shape[0]
    k = max(1, int(n * args.fraction))
    idx = torch.randperm(n)[:k]
    X = X[idx].numpy()

    out = run_sklearn_lp(X, n_classes=args.classes, iters=args.iters)
    print("[sklearn]", out)

    out = run_sknetwork_lp(X, n_classes=args.classes, iters=args.iters)
    print("[sknetwork]", out)

    out = run_pyg_lp(X, n_classes=args.classes, iters=args.iters, device=args.device)
    print("[PyG]", out)

if __name__ == "__main__":
    main()
