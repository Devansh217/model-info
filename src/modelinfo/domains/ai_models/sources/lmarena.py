from __future__ import annotations

import io

import pandas as pd

from ....core.records import PartialRecord
from ....core.sources import Fetcher, drop_none

URL = "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/refs%2Fconvert%2Fparquet/text/latest/0000.parquet"
PAGE = "https://lmarena.ai/leaderboard/text"


class LMArenaSource:
    name = "lmarena"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def fetch_all(self, refresh: bool = False) -> list[PartialRecord]:
        raw, fetched_at = self.fetcher.get(URL, refresh=refresh)
        df = pd.read_parquet(io.BytesIO(raw))
        df = df[df["category"] == "overall"]
        out = []
        for row in df.itertuples(index=False):
            lic = row.license if isinstance(row.license, str) and row.license else None
            org = row.organization if isinstance(row.organization, str) and row.organization else None
            fields = drop_none({
                "arena_elo": round(float(row.rating), 1),
                "provider": org,
                "license": lic,
                "open_weights": (lic.strip().lower() != "proprietary") if lic else None,
            })
            out.append(PartialRecord(self.name, row.model_name, row.model_name, PAGE, fetched_at, fields))
        return out
