from __future__ import annotations

from .records import Field, PartialRecord


def merge(records: list[PartialRecord], priority: dict[str, list[str]],
          default_order: list[str]) -> dict[str, Field]:
    """For each field, take the value from the first source in its priority list that has it."""
    by_source = {r.source: r for r in records}
    names = {k for r in records for k in r.fields}
    out: dict[str, Field] = {}
    for f in sorted(names):
        for src in priority.get(f, default_order):
            r = by_source.get(src)
            if r is not None and f in r.fields:
                out[f] = Field(r.fields[f], r.source, r.url, r.fetched_at)
                break
    return out
