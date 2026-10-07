from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .domain import Domain, Enricher
from .merge import merge
from .records import PartialRecord, Record
from .sources import CatalogSource, SearchSource


class Engine:
    """Generic lookup pipeline, identical for every domain:

    1. catalogs: bulk-loaded once (per lookup / per CSV), query fuzzy-matched against all names
    2. searchers: queried per lookup, query fuzzy-matched against returned candidates
    3. enrichers: second hop keyed on a field found in step 1/2 (e.g. an ID or website)
    4. merge: per field, first source in the domain's priority list wins
    """

    def __init__(self, domain: Domain, catalogs: list[CatalogSource] | None = None,
                 enrichers: list[Enricher] | None = None, searchers: list[SearchSource] | None = None):
        self.domain = domain
        self.catalogs = catalogs or []
        self.searchers = searchers or []
        self.enrichers = enrichers or []

    def _load(self, refresh: bool) -> tuple[dict[str, list[PartialRecord]], dict[str, str]]:
        data, errors = {}, {}
        for src in self.catalogs:
            try:
                data[src.name] = src.fetch_all(refresh=refresh)
            except Exception as e:  # one bad source must not break the lookup
                errors[src.name] = f"{type(e).__name__}: {e}"
                warnings.warn(f"{self.domain.name}: skipping {src.name}: {errors[src.name]}",
                              UserWarning, stacklevel=3)
        return data, errors

    def _lookup(self, query: str, data, errors, refresh: bool) -> Record:
        d = self.domain
        errors = dict(errors)
        matched: list[PartialRecord] = []
        matches: dict[str, tuple[str, float]] = {}
        data = dict(data)
        for src in self.searchers:
            try:
                data[src.name] = src.search(query, refresh=refresh)
            except Exception as e:
                errors[src.name] = f"{type(e).__name__}: {e}"
        for src_name, recs in data.items():
            hit = d.matcher.best_match(query, [r.name for r in recs])
            if hit:
                idx, score = hit
                matched.append(recs[idx])
                matches[src_name] = (recs[idx].name, score)
        if not matched:
            names = [r.name for recs in data.values() for r in recs]
            raise d.not_found_cls(query, d.matcher.suggest(query, names))

        for enr in self.enrichers:
            key = next((r.fields[enr.trigger_field] for r in matched if r.fields.get(enr.trigger_field)), None)
            if not key:
                continue
            try:
                extra = enr.source.fetch_one(key, refresh=refresh)
                matched.append(extra)
                matches[enr.source.name] = (extra.name, 100.0)
            except Exception as e:
                errors[enr.source.name] = f"{type(e).__name__}: {e}"
        return d.record_cls(query=query, fields=merge(matched, d.priority, d.default_order),
                            matches=matches, errors=errors, field_names=list(d.fields))

    def lookup(self, query: str, refresh: bool = False) -> Record:
        data, errors = self._load(refresh)
        return self._lookup(query, data, errors, refresh)

    def _best_match_label(self, matches: dict[str, tuple[str, float]]) -> tuple[str, float]:
        for src in self.domain.default_order:
            if src in matches:
                return matches[src]
        return max(matches.values(), key=lambda m: m[1])

    def from_csv(self, path: str | Path, refresh: bool = False) -> pd.DataFrame:
        d = self.domain
        key = d.key_column
        src = pd.read_csv(path)
        if key not in src.columns:
            raise ValueError(f"CSV must have a '{key}' column")
        requested = [c for c in src.columns if c != key]
        unknown = [c for c in requested if c not in d.fields]
        if unknown:
            raise ValueError(f"Unknown field columns: {unknown}. Known fields: {d.fields}")
        fields = requested or list(d.fields)

        data, errors = self._load(refresh)
        rows = []
        for query in src[key].astype(str):
            row: dict = {key: query}
            try:
                rec = self._lookup(query, data, errors, refresh)
                values = rec.to_dict(fields)
                for f in fields:
                    row.update(d.row_values(f, values[f]))
                row["matched_as"], row["match_score"] = self._best_match_label(rec.matches)
                row["gaps"] = ", ".join(f for f in fields if values[f] is None)
            except d.not_found_cls as e:
                for f in fields:
                    row.update(d.row_values(f, None))
                row["matched_as"], row["match_score"] = None, None
                row["gaps"] = "NOT FOUND; suggestions: " + ", ".join(e.suggestions)
            rows.append(row)
        columns = [c for f in fields for c in d.columns_for(f)]
        return pd.DataFrame(rows, columns=[key, *columns, "matched_as", "match_score", "gaps"])
