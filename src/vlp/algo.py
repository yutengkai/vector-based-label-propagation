"""
High-level VLP runner.
"""
from __future__ import annotations

import time
import math
import numpy as np
import torch
from torch import Tensor

from .core import vlp, transform_embeddings, precompute_stats
from .utils.seed import set_all_seeds
from .utils.device import resolve_device_dtype

@torch.no_grad()
def run_vlp(features: Tensor, n_classes: int = 50, iters: int = 1000, seed: int = 42,
            normalize: bool = True, add_bias: bool = True, device: str | None = None, dtype: str = "float32"):
    """
    Run VLP on the provided feature matrix (n x d). Returns (elapsed_seconds, Y_T, meta_dict).
    """
    dev, torch_dtype = resolve_device_dtype(device, dtype)
    set_all_seeds(seed)
    V = features.to(device=dev, dtype=torch_dtype, non_blocking=False)
    n = V.shape[0]
    # Initialize labels: random one-hot across n_classes
    rng = torch.Generator(device=dev)
    rng.manual_seed(seed)
    labels = torch.randint(low=0, high=n_classes, size=(n,), generator=rng, device=dev)
    Y0 = torch.zeros((n, n_classes), dtype=torch_dtype, device=dev)
    Y0.scatter_(1, labels.view(-1, 1), 1.0)

    start = time.perf_counter()
    YT = vlp(V, Y0, iters=iters, normalize=normalize, add_bias=add_bias)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    meta = dict(
        n=n, d=V.shape[1], classes=n_classes, iters=iters,
        device=str(dev), dtype=str(torch_dtype),
        torch_version=torch.__version__,
    )
    return elapsed, YT, meta
