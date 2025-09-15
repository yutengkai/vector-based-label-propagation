from __future__ import annotations

import importlib
import json
import math
import time
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path

from ..algo import run_vlp
from ..baselines.sklearn_lp import run_sklearn_lp
from ..baselines.sknetwork_lp import run_sknetwork_lp
from ..baselines.pyg_lp import run_pyg_lp
from ..utils.seed import set_all_seeds
from ..utils.device import resolve_device_dtype

METHODS = ["VLP", "sklearn", "sknetwork", "pyg"]

def subsample_rows(X: torch.Tensor, fraction: float, seed: int = 42) -> torch.Tensor:
    n = X.shape[0]
    k = max(1, int(math.ceil(n * fraction)))
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k]
    return X[idx]

def run_suite(dataset_loader: str, fractions, classes: int, iters: int, repeats: int, seed: int,
              device: str, dtype: str, out_csv: Path):
    mod_name, func_name = dataset_loader.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    load_fn = getattr(mod, func_name)
    X_all, meta = load_fn("data")
    results = []
    for frac in fractions:
        X = subsample_rows(X_all, frac, seed=seed).cpu()
        n = X.shape[0]
        # Repeat timing and average
        # VLP
        v_times = []
        for r in range(repeats):
            t, _, m = run_vlp(X, n_classes=classes, iters=iters, seed=seed + r, device=device, dtype=dtype)
            v_times.append(t)
        results.append(dict(dataset=meta["name"], fraction=frac, nodes=n, method="VLP", time_s=float(np.mean(v_times)), success=True))
        # scikit-learn
        s_times = []
        ok = True
        for r in range(repeats):
            out = run_sklearn_lp(X.numpy(), n_classes=classes, iters=iters, seed=seed + r)
            if not out["success"]:
                ok = False
                break
            s_times.append(out["elapsed"])
        results.append(dict(dataset=meta["name"], fraction=frac, nodes=n, method="scikit-learn",
                            time_s=(float(np.mean(s_times)) if ok else None), success=ok))
        # scikit-network
        k_times = []
        ok = True
        for r in range(repeats):
            out = run_sknetwork_lp(X.numpy(), n_classes=classes, iters=iters, seed=seed + r)
            if not out["success"]:
                ok = False
                break
            k_times.append(out["elapsed"])
        results.append(dict(dataset=meta["name"], fraction=frac, nodes=n, method="scikit-network",
                            time_s=(float(np.mean(k_times)) if ok else None), success=ok))
        # PyG
        p_times = []
        ok = True
        for r in range(repeats):
            out = run_pyg_lp(X.numpy(), n_classes=classes, iters=iters, seed=seed + r, device=device)
            if not out["success"]:
                ok = False
                break
            p_times.append(out["elapsed"])
        results.append(dict(dataset=meta["name"], fraction=frac, nodes=n, method="PyG",
                            time_s=(float(np.mean(p_times)) if ok else None), success=ok))
    # Write CSV
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","fraction","nodes","method","time_s","success"])
        for r in results:
            w.writerow([r["dataset"], r["fraction"], r["nodes"], r["method"], ("" if r["time_s"] is None else r["time_s"]), ("1" if r["success"] else "X")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to experiment YAML")
    ap.add_argument("--out", type=str, default="outputs")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="float32")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    with open("configs/datasets.yaml", "r") as f:
        ds = yaml.safe_load(f)

    ds_key = cfg["dataset"]
    loader = ds[ds_key]["module"] + "." + ds[ds_key]["loader"]
    fractions = cfg.get("fractions", ds[ds_key]["fractions"])
    classes = cfg.get("classes", 50)
    iters = cfg.get("iters", 1000)
    repeats = cfg.get("repeats", 5)

    out_csv = Path(args.out) / f"{ds_key}_results.csv"
    run_suite(loader, fractions, classes, iters, repeats, seed=42, device=args.device, dtype=args.dtype, out_csv=out_csv)

if __name__ == "__main__":
    main()
