from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"

_lock = Lock()


def _read_all() -> list[dict[str, Any]]:
    if not ANNOTATIONS_FILE.exists():
        return []
    with ANNOTATIONS_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_annotations(limit: int = 100) -> list[dict[str, Any]]:
    with _lock:
        data = _read_all()
    if limit <= 0:
        return data
    return data[-limit:]


def save_annotation(record: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        data = _read_all()
        data.append(record)
        with ANNOTATIONS_FILE.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    return record
