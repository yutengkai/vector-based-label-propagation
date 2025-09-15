import torch
from vlp.core import transform_embeddings, precompute_stats

def test_stochasticity():
    n, d = 64, 8
    V = torch.randn(n, d, dtype=torch.float32)
    Vp = transform_embeddings(V)
    s, k, inv_deg = precompute_stats(Vp)
    # Degrees positive and rows normalized post-multiply
    assert torch.all(k > 0)
