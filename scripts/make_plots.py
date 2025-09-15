#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))
# Placeholder plotting script (runtime vs nodes). Optional.
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    for method in df["method"].unique():
        d = df[df["method"]==method]
        plt.figure()
        plt.plot(d["nodes"], d["time_s"], marker="o")
        plt.xlabel("Nodes")
        plt.ylabel("Time (s)")
        plt.title(method)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.out.replace(".png", f"_{method}.png"))
        else:
            plt.show()

if __name__ == "__main__":
    main()
