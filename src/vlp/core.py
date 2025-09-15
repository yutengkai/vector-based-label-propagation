"""
Core VLP primitives.
"""
from __future__ import annotations

import torch
from torch import Tensor

def transform_embeddings(V: Tensor, normalize: bool = True, add_bias: bool = True) -> Tensor:
    """
    Row-normalize embeddings and (optionally) augment with a bias term to obtain non-negative similarities.
    Returns V' with shape (n, d [+ 1]).
    """
    if normalize:
        V = torch.nn.functional.normalize(V, p=2, dim=1, eps=1e-12)
    if add_bias:
        ones = torch.ones((V.shape[0], 1), dtype=V.dtype, device=V.device)
        V = torch.cat([V, ones], dim=1) * (1.0 / (2.0 ** 0.5))
    return V

@torch.no_grad()
def precompute_stats(Vp: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute self_loop (s), degree (k), and inverse degree (inv_deg) for V'.
    Returns (s, k, inv_deg) with shapes: (n,), (n,), (n,)
    """
    s = (Vp * Vp).sum(dim=1)  # self-loop (row-wise squared norm)
    Vsum = Vp.sum(dim=0)      # sum of rows (d,)
    k = (Vp @ Vsum) - s       # degree = row-sum of V V^T with diag removed
    eps = torch.finfo(Vp.dtype).eps
    inv_deg = torch.clamp(k, min=eps).reciprocal()
    return s, k, inv_deg

@torch.no_grad()
def vlp_one_iter(Vp: Tensor, Y: Tensor, s: Tensor, inv_deg: Tensor) -> Tensor:
    """
    One vector-based LP iteration:
      Z = V^T Y
      W = V Z
      W = W - s * Y
      Y_next = inv_deg * W
    Broadcasting is row-wise for s and inv_deg.
    """
    Z = Vp.T @ Y                       # (d x c)
    W = Vp @ Z                         # (n x c)
    W = W - s[:, None] * Y             # remove self contribution
    Y_next = inv_deg[:, None] * W      # row-wise degree normalization
    return Y_next

@torch.no_grad()
def vlp(V: Tensor, Y0: Tensor, iters: int = 1000, normalize: bool = True, add_bias: bool = True) -> Tensor:
    """
    Run VLP for a fixed number of iterations starting from Y0.
    Returns Y_T.
    """
    Vp = transform_embeddings(V, normalize=normalize, add_bias=add_bias)
    s, k, inv_deg = precompute_stats(Vp)
    Y = Y0
    for _ in range(iters):
        if V.is_cuda:
            torch.cuda.synchronize()
        Y = vlp_one_iter(Vp, Y, s, inv_deg)
    if V.is_cuda:
        torch.cuda.synchronize()
    return Y
