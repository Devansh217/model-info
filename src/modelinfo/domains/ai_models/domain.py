from __future__ import annotations

from dataclasses import dataclass, field

from ...core.cache import Cache
from ...core.domain import Domain, Enricher
from ...core.engine import Engine
from ...core.records import NotFound, Record
from ...core.sources import Fetcher
from .matcher import MATCHER
from .sources.artificialanalysis import ArtificialAnalysisSource
from .sources.huggingface import HuggingFaceSource
from .sources.lmarena import LMArenaSource
from .sources.openrouter import OpenRouterSource

OR, AA, LM, HF = "openrouter", "artificialanalysis", "lmarena", "huggingface"
DEFAULT_ORDER = [OR, AA, LM, HF]

DEFAULT_FIELDS = [
    "name",
    "provider",
    "context_window",
    "max_output_tokens",
    "input_price_per_mtok",
    "output_price_per_mtok",
    "modalities",
    "release_date",
    "open_weights",
    "license",
    "arena_elo",
    "mmlu_pro",
    "gpqa",
    "swe_bench",
    "aa_intelligence_index",
    "output_speed_tps",
    "latency_ttft_s",
]

PRIORITY: dict[str, list[str]] = {
    "name": [OR, AA, LM],
    "provider": [OR, AA, LM],
    "context_window": [OR],
    "max_output_tokens": [OR],
    "input_price_per_mtok": [OR, AA],
    "output_price_per_mtok": [OR, AA],
    "modalities": [OR],
    "release_date": [OR],
    "open_weights": [HF, OR, LM],
    "license": [HF, OR, LM],
    "arena_elo": [LM],
    "mmlu_pro": [AA],
    "gpqa": [AA],
    "swe_bench": [AA],
    "aa_intelligence_index": [AA, OR],
    "output_speed_tps": [AA],
    "latency_ttft_s": [AA],
}


@dataclass
class ModelRecord(Record):
    field_names: list[str] = field(default_factory=lambda: list(DEFAULT_FIELDS))


class ModelNotFound(NotFound):
    pass


AI_MODELS = Domain(
    name="modelinfo",
    key_column="model",
    fields=DEFAULT_FIELDS,
    priority=PRIORITY,
    default_order=DEFAULT_ORDER,
    matcher=MATCHER,
    record_cls=ModelRecord,
    not_found_cls=ModelNotFound,
)


class ModelInfo(Engine):
    def __init__(self, cache: Fetcher | None = None, aa_api_key: str | None = None):
        cache = cache or Cache()
        super().__init__(
            AI_MODELS,
            catalogs=[
                OpenRouterSource(cache),
                ArtificialAnalysisSource(cache, aa_api_key),
                LMArenaSource(cache),
            ],
            enrichers=[Enricher("hugging_face_id", HuggingFaceSource(cache))],
        )
