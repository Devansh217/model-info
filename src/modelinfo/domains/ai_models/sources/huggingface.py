from __future__ import annotations

import json

from ....core.records import PartialRecord
from ....core.sources import Fetcher, drop_none

API = "https://huggingface.co/api/models/{id}"


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def fetch_one(self, repo_id: str, refresh: bool = False) -> PartialRecord:
        raw, fetched_at = self.fetcher.get(API.format(id=repo_id), refresh=refresh)
        m = json.loads(raw)
        card = m.get("cardData") or {}
        lic = card.get("license")
        if lic == "other":
            lic = card.get("license_name") or lic
        rid = m.get("id", repo_id)
        return PartialRecord(self.name, rid, rid, f"https://huggingface.co/{rid}", fetched_at,
                             drop_none({"open_weights": True, "license": lic}))
