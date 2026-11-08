"""Read recorded sessions (``metadata.json`` and ``records.csv``) row by row, without pandas.

Cells are kept as the strings in the CSV; the ``cell_*`` helpers parse them.
Empty cells mean "not recorded" (e.g. no human command while the model drives).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .files import read_json


def read_metadata(session_dir: Path) -> dict[str, object]:
    """The session's metadata, or an empty dict if it has none."""
    path = session_dir / "metadata.json"
    return read_json(path) if path.is_file() else {}


def read_records(session_dir: Path) -> list[dict[str, str]]:
    """One dict per recorded frame, in recording order."""
    path = session_dir / "records.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No records.csv in {session_dir}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell_text(row: dict[str, str], column: str) -> str:
    return (row.get(column) or "").strip()


def cell_float(row: dict[str, str], column: str) -> float | None:
    """A finite number, or None for an empty cell."""
    text = cell_text(row, column)
    if not text:
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{column} is not finite: {text!r}")
    return value


def cell_flag(row: dict[str, str], column: str) -> bool:
    """A 0/1 flag; an empty cell counts as 0."""
    text = cell_text(row, column).lower()
    if text in {"", "0", "0.0", "false"}:
        return False
    if text in {"1", "1.0", "true"}:
        return True
    raise ValueError(f"{column} is not a 0/1 flag: {text!r}")
