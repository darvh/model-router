"""Tiny HTTP + file cache helpers. Stdlib only."""
from __future__ import annotations

import json
import os
import time
import urllib.request

USER_AGENT = "model-router/0.1 (+https://github.com/darvh/model-router)"


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_json(url: str, timeout: int = 30):
    return json.loads(fetch_bytes(url, timeout).decode("utf-8"))


def read_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def write_json(path: str, data) -> None:
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def fetch_json_cached(url: str, path: str, max_age_hours=None, force: bool = False):
    """Fetch JSON with an on-disk cache. max_age_hours=None means cache forever."""
    if not force and os.path.exists(path):
        if max_age_hours is None:
            return read_json(path)
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_hours < max_age_hours:
            return read_json(path)
    data = fetch_json(url)
    write_json(path, data)
    return data


def fetch_text_cached(url: str, path: str, force: bool = False) -> str:
    if not force and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    text = fetch_bytes(url).decode("utf-8", "replace")
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return text
