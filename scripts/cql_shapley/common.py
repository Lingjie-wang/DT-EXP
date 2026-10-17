"""Small file helpers shared by the isolated CQL delayed experiment."""

import hashlib
import json
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with open(path, "a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def verify(root):
    root = Path(root)
    p = read(root / "protocol.json")
    for relative, expected in p["source_sha256"].items():
        if digest(root / "source" / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    return p
