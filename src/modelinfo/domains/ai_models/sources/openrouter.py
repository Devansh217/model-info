from __future__ import annotations

import json
from datetime import datetime, timezone

from ....core.records import PartialRecord
from ....core.sources import Fetcher, drop_none

URL = "https://openrouter.ai/api/v1/models"


class OpenRouterSource:
    name = "openrouter"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def fetch_all(self, refresh: bool = False) -> list[PartialRecord]:
        raw, fetched_at = self.fetcher.get(URL, refresh=refresh)
        out = []
        for m in json.loads(raw)["data"]:
            mid = m["id"]
            if mid.startswith("~") or ":" in mid:
                continue
            out.append(PartialRecord(self.name, mid, m["name"], f"https://openrouter.ai/{mid}",
                                     fetched_at, _fields(m)))
        return out


def _per_mtok(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v * 1_000_000, 4) if v >= 0 else None


def _modalities(arch: dict) -> str | None:
    ins, outs = arch.get("input_modalities") or [], arch.get("output_modalities") or []
    if ins or outs:
        return f"{','.join(ins)}->{','.join(outs)}"
    return arch.get("modality")


def _fields(m: dict) -> dict:
    arch = m.get("architecture") or {}
    top = m.get("top_provider") or {}
    pricing = m.get("pricing") or {}
    aa = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
    name = m.get("name", "")
    provider = name.split(": ", 1)[0] if ": " in name else m["id"].split("/", 1)[0]
    created = m.get("created")
    hf_id = m.get("hugging_face_id") or None
    return drop_none({
        "name": name.split(": ", 1)[-1],
        "provider": provider,
        "context_window": m.get("context_length"),
        "max_output_tokens": top.get("max_completion_tokens"),
        "input_price_per_mtok": _per_mtok(pricing.get("prompt")),
        "output_price_per_mtok": _per_mtok(pricing.get("completion")),
        "modalities": _modalities(arch),
        "release_date": datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat() if created else None,
        "open_weights": True if hf_id else None,
        "hugging_face_id": hf_id,
        "aa_intelligence_index": aa.get("intelligence_index"),
    })
