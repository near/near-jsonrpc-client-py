import json
from pathlib import Path

import httpx


def load_openapi(source: str) -> dict:
    if source.startswith("https://") or source.startswith("https://"):
        return _load_from_url(source)
    else:
        return _load_from_file(source)


def _load_from_file(path: str) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_from_url(url: str) -> dict:
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
