"""Model catalog.

Costs and capabilities from models.dev (open, no key).
Optional benchmark indices from the Artificial Analysis free API (x-api-key, 100 req/day);
the response shape is not pinned here on purpose, indices are extracted tolerantly and
stored raw alongside so a shape change degrades to "no indices", never a crash.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ._net import fetch_json_cached, read_json, write_json

MODELS_DEV_URL = "https://models.dev/api.json"
AA_URL = "https://artificialanalysis.ai/api/v2/language/models/free"  # free tier endpoint
AA_CACHE = "aa-language-models.json"
AA_INDICES_CACHE = "aa-indices.json"


@dataclass
class Model:
    ref: str  # "openai/gpt-5.6-luna"
    provider: str
    id: str
    name: str
    cost_in: float  # USD per 1M input tokens
    cost_out: float  # USD per 1M output tokens
    context: int
    tool_call: bool
    reasoning: bool
    released: str
    indices: Optional[Dict[str, float]] = None
    reasoning_options: Optional[List[dict]] = None

    @property
    def blended_cost(self) -> float:
        """Rough ranking cost: mostly output-weighted."""
        return (self.cost_in + 3.0 * self.cost_out) / 4.0


def load_models(cache_dir: str, force: bool = False) -> Dict[str, Model]:
    """Load models.dev catalog keyed by 'provider/id'."""
    raw = fetch_json_cached(
        MODELS_DEV_URL, os.path.join(cache_dir, "models-dev.json"), max_age_hours=24, force=force
    )
    models: Dict[str, Model] = {}
    for provider_id, provider in raw.items():
        for model_id, m in (provider.get("models") or {}).items():
            cost = m.get("cost") or {}
            if cost.get("input") is None or cost.get("output") is None:
                continue
            ref = "%s/%s" % (provider_id, model_id)
            models[ref] = Model(
                ref=ref,
                provider=provider_id,
                id=model_id,
                name=m.get("name") or model_id,
                cost_in=float(cost["input"]),
                cost_out=float(cost["output"]),
                context=int((m.get("limit") or {}).get("context") or 0),
                tool_call=bool(m.get("tool_call")),
                reasoning=bool(m.get("reasoning")),
                released=str(m.get("release_date") or ""),
                reasoning_options=list(m.get("reasoning_options") or []),
            )
    return models


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


# Prefer lab-direct providers over aggregators/gateways when resolving trial refs.
PREFERRED_PROVIDERS = (
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "moonshotai",
    "zhipuai",
    "xai",
    "alibaba",
    "minimax",
    "meta",
    "mistral",
)


def canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def build_model_index(models: Dict[str, "Model"]) -> Dict[str, List[str]]:
    """Index models by canonical id (ignores '.' vs '-' vs '_')."""
    idx: Dict[str, List[str]] = {}
    for ref, m in models.items():
        idx.setdefault(canon(m.id), []).append(ref)
    return idx


def resolve_ref(provider: str, model: str, models: Dict[str, "Model"], index: Optional[Dict] = None) -> Optional[str]:
    """Map a trial (provider, model) onto a models.dev ref, tolerating naming drift.

    Trial providers (vertex_ai, zai, moonshot, gemini, ...) name the serving gateway;
    we resolve to the lab-direct ref the config uses when one exists.
    """
    if not model:
        return None
    index = index if index is not None else build_model_index(models)
    candidates = index.get(canon(model))
    if not candidates:
        return None
    for pref in PREFERRED_PROVIDERS:
        for ref in candidates:
            if ref.split("/", 1)[0] == pref:
                return ref
    return candidates[0]


def load_aa_indices(
    cache_dir: str,
    api_key: Optional[str] = None,
    force: bool = False,
    max_age_hours: float = 168.0,
    raise_errors: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Artificial Analysis index scores keyed by canon(slug).

    Uses the free endpoint (paginated, ~4 requests), cached for a week by default.
    Without a key, returns the cached copy (possibly empty); DeepSWE quality still works.
    """
    indices_path = os.path.join(cache_dir, AA_INDICES_CACHE)
    cached = read_json(indices_path)
    if not force and cached is not None:
        age_hours = (time.time() - os.path.getmtime(indices_path)) / 3600.0
        if age_hours < max_age_hours:
            return cached
    key = api_key or os.environ.get("AA_API_KEY")
    if not key:
        return cached or {}
    try:
        entries: List[dict] = []
        page = 1
        while True:
            data = fetch_json_with_key("%s?page=%d" % (AA_URL, page), key)
            entries.extend(data.get("data") or [])
            pagination = data.get("pagination") or {}
            if not pagination.get("has_more"):
                break
            page += 1
            if page > 10:
                raise RuntimeError("AA pagination exceeded 10 pages; refusing to cache partial data")
    except Exception:
        if raise_errors and not cached:
            raise
        return cached or {}
    write_json(os.path.join(cache_dir, AA_CACHE), entries)
    out = _extract_aa_entries(entries)
    write_json(indices_path, out)
    return out


def fetch_json_with_key(url: str, key: str, timeout: int = 30):
    import json
    import urllib.request

    from ._net import USER_AGENT

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "x-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def tier_of_ref(ref: str, config: dict, models: Dict[str, Model]) -> Optional[str]:
    """Tier for a model ref: explicit tier membership first, then cost bands."""
    tiers = config["tiers"]
    for tier, spec in tiers.items():
        if ref in spec["models"]:
            return tier
    m = models.get(ref)
    if not m:
        return None
    bands = config.get("cost_bands") or {}
    if m.cost_out <= float(bands.get("utility", 1.5)):
        return "utility"
    if m.cost_out <= float(bands.get("balanced", 15.0)):
        return "balanced"
    return "frontier"


AA_QUALITY_FIELDS = ("coding", "agentic", "intelligence")


def quality_from_indices(indices: Optional[Dict[str, float]]) -> Optional[float]:
    """Capability 0..1 from AA index scores (coding > agentic > intelligence).

    AA free-endpoint indices are on a 0..100 scale (median intelligence ~12).
    """
    if not indices:
        return None
    for field in AA_QUALITY_FIELDS:
        for key, value in indices.items():
            if "cost" in key or "price" in key:
                continue
            if field in key and isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0.0, min(1.0, float(value) / 100.0))
    return None


def _extract_aa_entries(entries) -> Dict[str, Dict[str, float]]:
    """AA free-endpoint entries -> {canon(slug): {index_name: value}}."""
    out: Dict[str, Dict[str, float]] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug") or entry.get("name")
        evaluations = entry.get("evaluations")
        if not slug or not isinstance(evaluations, dict):
            continue
        nums = {
            str(k).lower(): float(v)
            for k, v in evaluations.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and "cost" not in str(k).lower()
            and "price" not in str(k).lower()
        }
        if nums:
            out[canon(str(slug))] = nums
    return out


def rank_candidates(
    refs: List[str],
    models: Dict[str, Model],
    quality: Optional[Dict[str, float]] = None,
    costs: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float, Optional[float]]]:
    """Rank candidate refs by quality/cost value. Returns [(ref, cost, quality)].

    quality: optional ref -> score (e.g. DeepSWE pass rate)
    costs:   optional ref -> cost override (e.g. measured $/task); else blended token price
    Without quality, sorts by cost ascending; unmeasured refs (quality None) sink to the end.
    """
    quality = quality or {}
    costs = costs or {}
    rows = []
    for ref in refs:
        m = models.get(ref)
        if not m:
            continue
        q = quality.get(ref)
        c = costs.get(ref)
        rows.append((ref, float(c) if c is not None else m.blended_cost, q))
    if any(q is not None for _, _, q in rows):
        rows.sort(key=lambda r: (-(r[2] or 0.0) / max(r[1], 1e-9), r[1]))
    else:
        rows.sort(key=lambda r: r[1])
    return rows
