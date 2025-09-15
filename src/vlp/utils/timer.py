from contextlib import contextmanager
import time
import torch

@contextmanager
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"[timing] {end - start:.6f}s")
