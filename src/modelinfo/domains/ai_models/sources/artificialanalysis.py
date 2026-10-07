from __future__ import annotations

import json
import os

from ....core.records import PartialRecord
from ....core.sources import Fetcher, SourceUnavailable, drop_none

URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


class ArtificialAnalysisSource:
    name = "artificialanalysis"

    def __init__(self, fetcher: Fetcher, api_key: str | None = None):
        self.fetcher = fetcher
        self.api_key = api_key or os.environ.get("AA_API_KEY")

    def fetch_all(self, refresh: bool = False) -> list[PartialRecord]:
        if not self.api_key:
            raise SourceUnavailable("AA_API_KEY not set (free key: https://artificialanalysis.ai/api-reference)")
        raw, fetched_at = self.fetcher.get(URL, headers={"x-api-key": self.api_key}, refresh=refresh)
        out = []
        for m in json.loads(raw)["data"]:
            ev = m.get("evaluations") or {}
            pr = m.get("pricing") or {}
            swe = ev.get("swe_bench")
            if swe is None:
                swe = ev.get("swe_bench_verified")
            fields = drop_none({
                "name": m.get("name"),
                "provider": (m.get("model_creator") or {}).get("name"),
                "aa_intelligence_index": ev.get("artificial_analysis_intelligence_index"),
                "mmlu_pro": ev.get("mmlu_pro"),
                "gpqa": ev.get("gpqa"),
                "swe_bench": swe,
                "input_price_per_mtok": pr.get("price_1m_input_tokens"),
                "output_price_per_mtok": pr.get("price_1m_output_tokens"),
                "output_speed_tps": m.get("median_output_tokens_per_second") or None,
                "latency_ttft_s": m.get("median_time_to_first_token_seconds") or None,
            })
            url = f"https://artificialanalysis.ai/models/{m.get('slug', '')}"
            out.append(PartialRecord(self.name, m["id"], m["name"], url, fetched_at, fields))
        return out
