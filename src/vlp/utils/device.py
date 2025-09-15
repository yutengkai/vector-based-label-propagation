from __future__ import annotations
import torch

def resolve_device_dtype(device: str | None, dtype: str = "float32"):
    if device in (None, "auto"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    dt_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dev, dt_map.get(dtype, torch.float32)
