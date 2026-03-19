"""
Helpers — Utility functions dùng chung.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file, trả về dict rỗng nếu file không tồn tại."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file (auto-create dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Get env variable with optional validation."""
    value = os.environ.get(key, default)
    if required and not value:
        raise EnvironmentError(f"Required env variable '{key}' is not set.")
    return value or ""


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
