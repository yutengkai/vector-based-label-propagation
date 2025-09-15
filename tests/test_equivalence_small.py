import torch
import numpy as np
from vlp.core import vlp, transform_embeddings, precompute_stats, vlp_one_iter

def test_equivalence_small():
    n, d, c = 128, 16, 3
    V = torch.randn(n, d, dtype=torch.float32)
    Vp = transform_embeddings(V)
    s, k, inv_deg = precompute_stats(Vp)
    Y = torch.zeros(n, c)
    labels = torch.randint(0, c, (n,))
    Y.scatter_(1, labels[:,None], 1.0)

    # VLP update
    Y1 = vlp_one_iter(Vp, Y, s, inv_deg)

    # Explicit adjacency update (for small n)
    A = Vp @ Vp.T
    A.fill_diagonal_(0.0)
    deg = A.sum(dim=1)
    inv = (deg.clamp_min(1e-12)).reciprocal()
    Y1_ref = inv[:,None] * (A @ Y)

    assert torch.allclose(Y1, Y1_ref, atol=1e-5)
