#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse
import yaml
from pathlib import Path
from vlp.eval.runner import run_suite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
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
    print(f"[benchmark] wrote {out_csv}")

if __name__ == "__main__":
    main()
