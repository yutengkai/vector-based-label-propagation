"""
scikit-learn LabelPropagation baseline with precomputed dense kernel (normalized cosine in [0,1], zero diag).
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.exceptions import ConvergenceWarning
import warnings

from ..utils.memory import dense_feasible
from ..utils.seed import set_all_seeds

def cosine_kernel_01(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    K = (Xn @ Xn.T + 1.0) * 0.5
    np.fill_diagonal(K, 0.0)
    return K

def run_sklearn_lp(features: np.ndarray, n_classes: int = 50, iters: int = 1000, seed: int = 42):
    n = features.shape[0]
    if not dense_feasible(n):
        return dict(success=False, reason="OOM_guard", elapsed=None, meta=dict(n=n))

    set_all_seeds(seed)

    start = time.perf_counter()
    K = cosine_kernel_01(features.astype(np.float64, copy=False))
    kernel_time = time.perf_counter() - start

    labels = np.random.RandomState(seed).randint(0, n_classes, size=(n,))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf = LabelPropagation(kernel="precomputed", max_iter=iters)
        start = time.perf_counter()
        clf.fit(K, labels)
        fit_time = time.perf_counter() - start

    elapsed = kernel_time + fit_time
    meta = dict(n=n, kernel_time_s=kernel_time, fit_time_s=fit_time, iters=iters)
    return dict(success=True, elapsed=elapsed, meta=meta)
