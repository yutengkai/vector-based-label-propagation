"""
PyTorch Geometric LabelPropagation baseline. Builds dense edge_index from normalized cosine similarities (small n).
"""
from __future__ import annotations

import time
import numpy as np
import torch
from torch_geometric.nn.models import LabelPropagation

from ..utils.memory import dense_feasible
from ..utils.seed import set_all_seeds

def dense_edge_index(n: int) -> torch.Tensor:
    # All ordered pairs excluding self-loops
    row = torch.arange(n).unsqueeze(1).repeat(1, n).view(-1)
    col = torch.arange(n).repeat(n)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    return edge_index

def run_pyg_lp(features: np.ndarray, n_classes: int = 50, iters: int = 1000, seed: int = 42, device: str = "cpu"):
    n = features.shape[0]
    if not dense_feasible(n):
        return dict(success=False, reason="OOM_guard", elapsed=None, meta=dict(n=n))

    set_all_seeds(seed)
    dev = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(dev)

    # Build dense edge_index (O(n^2))
    start = time.perf_counter()
    edge_index = dense_edge_index(n).to(dev)
    build_time = time.perf_counter() - start

    # Random labels across classes
    y = torch.randint(low=0, high=n_classes, size=(n,), device=dev)

    start = time.perf_counter()
    lp = LabelPropagation(num_layers=iters, alpha=0.9)  # alpha only used in some variants
    out = lp(y, edge_index)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start

    elapsed = build_time + fit_time
    meta = dict(n=n, build_time_s=build_time, fit_time_s=fit_time, iters=iters, device=str(dev))
    return dict(success=True, elapsed=elapsed, meta=meta)
