from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PartialRecord:
    """What one source knows about one entity."""

    source: str
    source_id: str
    name: str
    url: str
    fetched_at: str
    fields: dict[str, Any]


@dataclass
class Field:
    value: Any
    source: str
    url: str
    fetched_at: str


@dataclass
class Record:
    """Merged view of one entity across all sources. `field_names` is the domain's field list."""

    query: str
    fields: dict[str, Field]
    matches: dict[str, tuple[str, float]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    field_names: list[str] = field(default_factory=list)

    @property
    def gaps(self) -> list[str]:
        return [f for f in self.field_names if f not in self.fields]

    @property
    def sources(self) -> dict[str, str]:
        return {k: v.source for k, v in self.fields.items()}

    def to_dict(self, fields: list[str] | None = None) -> dict[str, Any]:
        wanted = fields or self.field_names
        return {f: (self.fields[f].value if f in self.fields else None) for f in wanted}

    def table(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "field": f,
                    "value": self.fields[f].value if f in self.fields else None,
                    "source": self.fields[f].source if f in self.fields else None,
                }
                for f in self.field_names
            ]
        )


class NotFound(LookupError):
    def __init__(self, query: str, suggestions: list[str]):
        self.query = query
        self.suggestions = suggestions
        hint = ", ".join(suggestions) or "n/a"
        super().__init__(f"No source matched {query!r}. Did you mean: {hint}?")
