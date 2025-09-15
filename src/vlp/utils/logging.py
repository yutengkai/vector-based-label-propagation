from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

def write_jsonl(path: str, record: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({**record, "ts": datetime.utcnow().isoformat()}) + "\n")
