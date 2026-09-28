import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_torch(path):
    # Checkpoints contain tensors/primitives only; never enable arbitrary pickle objects.
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_digest(items):
    sha = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        sha.update(name.encode())
        sha.update(str((tuple(value.shape), value.dtype)).encode())
        sha.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def require_empty(path):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output path.")
