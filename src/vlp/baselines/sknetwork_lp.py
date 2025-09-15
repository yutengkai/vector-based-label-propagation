"""
scikit-network Propagation baseline given CSR adjacency built from dense normalized cosine similarities in [0,1].
"""
from __future__ import annotations

import time
import numpy as np
from scipy.sparse import csr_matrix
from sknetwork.classification import Propagation

from ..utils.memory import dense_feasible
from ..utils.seed import set_all_seeds

def cosine_dense_to_csr(X: np.ndarray) -> csr_matrix:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    K = (Xn @ Xn.T + 1.0) * 0.5
    np.fill_diagonal(K, 0.0)
    # Build CSR from dense (small n only)
    rows, cols = np.nonzero(K > 0)
    data = K[rows, cols]
    return csr_matrix((data, (rows, cols)), shape=K.shape)

def run_sknetwork_lp(features: np.ndarray, n_classes: int = 50, iters: int = 1000, seed: int = 42):
    n = features.shape[0]
    if not dense_feasible(n):
        return dict(success=False, reason="OOM_guard", elapsed=None, meta=dict(n=n))

    set_all_seeds(seed)
    start = time.perf_counter()
    A = cosine_dense_to_csr(features.astype(np.float64, copy=False))
    build_time = time.perf_counter() - start

    labels = np.random.RandomState(seed).randint(0, n_classes, size=(n,))
    # Propagation expects -1 for unlabeled; for timing consistency we keep all labeled.
    start = time.perf_counter()
    clf = Propagation(n_iter=iters, damping_factor=0.0, verbose=False)
    clf.fit(A, labels=labels)
    fit_time = time.perf_counter() - start

    elapsed = build_time + fit_time
    meta = dict(n=n, build_time_s=build_time, fit_time_s=fit_time, iters=iters)
    return dict(success=True, elapsed=elapsed, meta=meta)
