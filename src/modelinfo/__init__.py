from __future__ import annotations

from pathlib import Path

from .core import Field
from .domains.ai_models import DEFAULT_FIELDS, ModelInfo, ModelNotFound, ModelRecord

__all__ = ["lookup", "from_csv", "ModelInfo", "ModelRecord", "ModelNotFound", "Field", "DEFAULT_FIELDS"]

_default: ModelInfo | None = None


def _client() -> ModelInfo:
    global _default
    if _default is None:
        _default = ModelInfo()
    return _default


def lookup(query: str, refresh: bool = False) -> ModelRecord:
    return _client().lookup(query, refresh=refresh)


def from_csv(path: str | Path, refresh: bool = False):
    return _client().from_csv(path, refresh=refresh)
