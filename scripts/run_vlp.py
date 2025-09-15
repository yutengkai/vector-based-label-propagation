#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
import yaml
import torch
from vlp.data.flickr import load_flickr
from vlp.algo import run_vlp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="flickr", choices=["flickr","amazon","yelp","taobao"])
    ap.add_argument("--fraction", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="float32")
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
    X = X[idx]

    elapsed, YT, meta_run = run_vlp(X, n_classes=args.classes, iters=args.iters, device=args.device, dtype=args.dtype)
    print(f"[VLP] dataset={args.dataset} nodes={X.shape[0]} d={X.shape[1]} iters={args.iters} classes={args.classes} time_s={elapsed:.4f}")
    print(meta_run)

if __name__ == "__main__":
    main()
