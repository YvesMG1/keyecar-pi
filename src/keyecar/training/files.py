"""Small file helpers: JSON reading and writing, and SHA-256 hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path: Path | str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, data: object, *, sort_keys: bool = False, allow_nan: bool = True) -> None:
    """Write indented JSON with a trailing newline."""
    text = json.dumps(data, indent=2, sort_keys=sort_keys, allow_nan=allow_nan)
    Path(path).write_text(text + "\n", encoding="utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
