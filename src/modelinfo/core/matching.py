from __future__ import annotations

import re
from typing import Callable, Protocol

from rapidfuzz import fuzz, process


class Matcher(Protocol):
    def best_match(self, query: str, candidates: list[str]) -> tuple[int, float] | None: ...

    def suggest(self, query: str, candidates: list[str], limit: int = 3) -> list[str]: ...


def basic_normalise(name: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", name.lower()).split())


class FuzzyMatcher:
    """Token-set fuzzy matcher. Domains plug in their own `normalise` and an optional
    `compatible(norm_query, norm_candidate)` veto (e.g. version numbers must agree)."""

    def __init__(self, normalise: Callable[[str], str] = basic_normalise,
                 compatible: Callable[[str, str], bool] | None = None, threshold: float = 85):
        self.normalise = normalise
        self.compatible = compatible
        self.threshold = threshold

    def score(self, query: str, candidate: str) -> float:
        q, c = self.normalise(query), self.normalise(candidate)
        if not q or not c or (self.compatible and not self.compatible(q, c)):
            return 0.0
        return float(fuzz.token_set_ratio(q, c))

    def best_match(self, query: str, candidates: list[str]) -> tuple[int, float] | None:
        nq = self.normalise(query)
        best = None
        for i, cand in enumerate(candidates):
            s = self.score(query, cand)
            if s < self.threshold:
                continue
            key = (s, fuzz.ratio(nq, self.normalise(cand)))
            if best is None or key > best[0]:
                best = (key, i)
        return (best[1], best[0][0]) if best else None

    def suggest(self, query: str, candidates: list[str], limit: int = 3) -> list[str]:
        uniq = list(dict.fromkeys(candidates))
        hits = process.extract(query, uniq, scorer=fuzz.token_set_ratio, processor=self.normalise, limit=limit)
        return [c for c, _, _ in hits]
