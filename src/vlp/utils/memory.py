from __future__ import annotations
import os
import psutil

# Heuristic guard for dense n x n allocations in baselines
def dense_feasible(n: int, bytes_per: int = 8, headroom: float = 0.6) -> bool:
    # K requires n^2 entries of 8 bytes if float64
    need = n * n * bytes_per
    try:
        avail = psutil.virtual_memory().available
    except Exception:
        avail = 4 * (1024 ** 3)  # assume 4GB available
    return need < headroom * avail and n <= 20000  # hard cap for safety
