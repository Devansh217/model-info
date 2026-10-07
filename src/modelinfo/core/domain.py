from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .matching import FuzzyMatcher, Matcher
from .records import NotFound, Record
from .sources import DetailSource

# Turns one (possibly nested) field value into flat CSV columns. Must return the same keys
# for every input, including None, so the output table has stable columns.
Flattener = Callable[[Any], dict[str, Any]]


def repeated(n: int, columns: dict[str, str]) -> Flattener:
    """Flatten a list of dicts into n numbered column groups.

    repeated(4, {"name": "Director {i} Name", "phone": "Director {i} Phone"})
    turns [{"name": "A", "phone": "1"}] into {"Director 1 Name": "A", "Director 1 Phone": "1",
    "Director 2 Name": None, ...}.
    """

    def flatten(value: Any) -> dict[str, Any]:
        items = list(value or [])[:n]
        out: dict[str, Any] = {}
        for i in range(1, n + 1):
            item = items[i - 1] if i <= len(items) else {}
            for key, template in columns.items():
                out[template.format(i=i)] = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        return out

    return flatten


@dataclass
class Domain:
    """Declarative description of what is being looked up. Everything domain-specific lives here
    (plus the source classes), so the engine never changes when a new domain is added."""

    name: str
    key_column: str                     # input CSV column holding the lookup query
    fields: list[str]                   # output fields, in display order
    priority: dict[str, list[str]]      # field -> sources to try, best first
    default_order: list[str]            # source order for fields missing from `priority`
    matcher: Matcher = field(default_factory=FuzzyMatcher)
    record_cls: type[Record] = Record
    not_found_cls: type[NotFound] = NotFound
    flatten: dict[str, Flattener] = field(default_factory=dict)  # field -> CSV column expansion

    def columns_for(self, f: str) -> list[str]:
        return list(self.flatten[f](None)) if f in self.flatten else [f]

    def row_values(self, f: str, value: Any) -> dict[str, Any]:
        return self.flatten[f](value) if f in self.flatten else {f: value}


@dataclass
class Enricher:
    """Second hop: if a matched record has `trigger_field`, fetch `source.fetch_one(value)`."""

    trigger_field: str
    source: DetailSource
