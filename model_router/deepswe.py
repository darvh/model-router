"""DeepSWE benchmark data (datacurve).

Downloads the public artifacts (tasks, trial outcomes, per-task instruction texts)
and derives calibration labels + evaluation stats:

  label for a task = cheapest tier whose model ever (or reliably) passed it,
  fallback frontier when nothing passed.

Endpoints discovered from the data browser frontend:
  https://deepswe.datacurve.ai/artifacts/v1.1/tasks.json
  https://deepswe.datacurve.ai/artifacts/v1.1/trials.json
  https://deepswe.datacurve.ai/artifacts/v1.1/tasks/<task-id>.json   (contains instruction.md)
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from ._net import ensure_dir, fetch_json_cached, fetch_text_cached

BASE = "https://deepswe.datacurve.ai/artifacts"
DEFAULT_VERSION = "v1.1"
TIERS = ("utility", "balanced", "frontier")
TIER_RANK = {t: i for i, t in enumerate(TIERS)}


def tasks_url(version: str = DEFAULT_VERSION) -> str:
    return "%s/%s/tasks.json" % (BASE, version)


def trials_url(version: str = DEFAULT_VERSION) -> str:
    return "%s/%s/trials.json" % (BASE, version)


def task_url(version: str, task_id: str) -> str:
    return "%s/%s/tasks/%s.json" % (BASE, version, task_id)


def _data_dir(cache_dir: str, version: str) -> str:
    return os.path.join(cache_dir, "deepswe", version)


def fetch_tasks(cache_dir: str, version: str = DEFAULT_VERSION, force: bool = False) -> List[dict]:
    path = os.path.join(_data_dir(cache_dir, version), "tasks.json")
    data = fetch_json_cached(tasks_url(version), path, max_age_hours=None, force=force)
    return data["rows"]


def fetch_trials(cache_dir: str, version: str = DEFAULT_VERSION, force: bool = False) -> List[dict]:
    path = os.path.join(_data_dir(cache_dir, version), "trials.json")
    data = fetch_json_cached(trials_url(version), path, max_age_hours=None, force=force)
    return data["rows"]


def _instruction_from_detail(detail: dict) -> str:
    for f in detail.get("files") or []:
        if f.get("path") == "instruction.md":
            return f.get("content") or ""
    return ""


def fetch_instructions(
    cache_dir: str,
    task_ids: List[str],
    version: str = DEFAULT_VERSION,
    force: bool = False,
    workers: int = 8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, str]:
    """Fetch instruction.md text per task, cached individually."""
    out_dir = os.path.join(_data_dir(cache_dir, version), "instructions")

    def one(task_id: str) -> Tuple[str, str]:
        path = os.path.join(out_dir, task_id + ".md")
        if not force and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return task_id, f.read()
        text = fetch_text_cached(task_url(version, task_id), path)
        try:
            text = _instruction_from_detail(json.loads(text))
        except Exception:
            text = ""
        ensure_dir(path)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
        return task_id, text

    results: Dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for task_id, text in ex.map(one, task_ids):
            results[task_id] = text
            done += 1
            if progress:
                progress(done, len(task_ids))
    return results


def filter_source(trials: List[dict], source: str = "deep-swe") -> List[dict]:
    return [t for t in trials if t.get("source") == source]


def model_stats(trials: List[dict], source: str = "deep-swe") -> Dict[str, dict]:
    """Aggregate per model ref: pass rate, avg cost over non-errored trials."""
    acc: Dict[str, dict] = {}
    for t in filter_source(trials, source):
        ref = "%s/%s" % (t.get("provider"), t.get("model"))
        a = acc.setdefault(ref, {"n": 0, "passed": 0, "errored": 0, "cost_sum": 0.0, "cost_n": 0})
        a["n"] += 1
        if t.get("errored"):
            a["errored"] += 1
        else:
            if t.get("passed"):
                a["passed"] += 1
            c = t.get("cost_usd")
            if isinstance(c, (int, float)):
                a["cost_sum"] += float(c)
                a["cost_n"] += 1
    out = {}
    for ref, a in acc.items():
        ok = a["n"] - a["errored"]
        out[ref] = {
            "n": a["n"],
            "pass_rate": (a["passed"] / ok) if ok > 0 else None,
            "avg_cost_usd": (a["cost_sum"] / a["cost_n"]) if a["cost_n"] else None,
        }
    return out


def task_model_stats(trials: List[dict], source: str = "deep-swe") -> Dict[Tuple[str, str], dict]:
    acc: Dict[Tuple[str, str], dict] = {}
    for t in filter_source(trials, source):
        key = (t.get("task_name"), "%s/%s" % (t.get("provider"), t.get("model")))
        a = acc.setdefault(key, {"n": 0, "passed": 0, "errored": 0, "cost_sum": 0.0, "cost_n": 0})
        a["n"] += 1
        if t.get("errored"):
            a["errored"] += 1
        else:
            if t.get("passed"):
                a["passed"] += 1
            c = t.get("cost_usd")
            if isinstance(c, (int, float)):
                a["cost_sum"] += float(c)
                a["cost_n"] += 1
    out = {}
    for key, a in acc.items():
        ok = a["n"] - a["errored"]
        out[key] = {
            "n": a["n"],
            "pass_rate": (a["passed"] / ok) if ok > 0 else None,
            "avg_cost_usd": (a["cost_sum"] / a["cost_n"]) if a["cost_n"] else None,
        }
    return out


def labels(
    tasks: List[dict],
    trials: List[dict],
    tier_of: Callable[[str], Optional[str]],
    min_rate: float = 0.5,
    source: str = "deep-swe",
) -> Dict[str, str]:
    """Cheapest tier whose model reliably passed each task (fallback: any pass, then frontier)."""
    per_task: Dict[str, Dict[str, dict]] = {}
    for t in filter_source(trials, source):
        task = t.get("task_name")
        if task is None:
            continue
        ref = "%s/%s" % (t.get("provider"), t.get("model"))
        a = per_task.setdefault(task, {}).setdefault(ref, {"ok": 0, "passed": 0})
        if not t.get("errored"):
            a["ok"] += 1
            if t.get("passed"):
                a["passed"] += 1

    out: Dict[str, str] = {}
    for task_row in tasks:
        task_id = task_row.get("id")
        models = per_task.get(task_id, {})
        reliable: List[str] = []
        any_pass: List[str] = []
        for ref, a in models.items():
            tier = tier_of(ref)
            if tier is None:
                tier = "frontier"  # unpriceable/unknown model: conservative
            if a["ok"] and a["passed"] / a["ok"] >= min_rate:
                reliable.append(tier)
            if a["passed"] > 0:
                any_pass.append(tier)
        if reliable:
            out[task_id] = min(reliable, key=lambda t: TIER_RANK[t])
        elif any_pass:
            out[task_id] = min(any_pass, key=lambda t: TIER_RANK[t])
        else:
            out[task_id] = "frontier"
    return out
