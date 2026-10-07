from __future__ import annotations

from typing import Protocol

from .records import PartialRecord


class SourceUnavailable(RuntimeError):
    """Source cannot be used (e.g. missing API key)."""


class Fetcher(Protocol):
    """Returns raw bytes + fetch timestamp. Sources parse the bytes themselves (JSON, HTML, parquet, ...)."""

    def get(self, url: str, headers: dict | None = None, refresh: bool = False) -> tuple[bytes, str]: ...


class RequestFetcher(Fetcher, Protocol):
    """Fetcher that also supports POST forms, query params and JSON bodies (e.g. registry search pages)."""

    def request(self, method: str, url: str, *, headers: dict | None = None, params: dict | None = None,
                data: dict | None = None, json: object = None, refresh: bool = False) -> tuple[bytes, str]: ...


class CatalogSource(Protocol):
    """Source that lists every entity it knows; the engine fuzzy-matches the query against them."""

    name: str

    def fetch_all(self, refresh: bool = False) -> list[PartialRecord]: ...


class SearchSource(Protocol):
    """Source that is queried per lookup (site search, registry search). The engine
    fuzzy-matches the query against the returned candidates."""

    name: str

    def search(self, query: str, refresh: bool = False) -> list[PartialRecord]: ...


class DetailSource(Protocol):
    """Source that fetches one entity by a key found by another source (see `Enricher`)."""

    name: str

    def fetch_one(self, key: str, refresh: bool = False) -> PartialRecord: ...


def drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v == v}  # v == v drops NaN
