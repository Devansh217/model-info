from __future__ import annotations

import re

from ...core.matching import FuzzyMatcher

THRESHOLD = 85

NOISE = {"high", "low", "medium", "thinking", "reasoning", "preview", "latest",
         "instruct", "chat", "exp", "experimental"}
VARIANTS = {"mini", "nano", "flash", "lite", "pro", "max", "small", "large", "turbo",
            "haiku", "sonnet", "opus", "plus", "ultra", "air", "fast", "codex", "coder", "vision"}

_VERSION = re.compile(r"\d+(?:\.\d+)*[a-z]*")
_DATE = re.compile(r"\b\d{4}[ -]?\d{2}[ -]?\d{2}\b|\b\d{8}\b")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def normalise(name: str) -> str:
    s = name.strip().lower()
    s = _PAREN.sub("", s)
    if ": " in s:
        s = s.split(": ", 1)[1]
    s = s.split(":", 1)[0]
    s = s.rsplit("/", 1)[-1]
    s = re.sub(r"[-_\s]+", " ", s)
    s = _DATE.sub(" ", s)
    s = re.sub(r"(?<=\d)\s+(?=\d\b)", ".", s)
    return " ".join(t for t in s.split() if t not in NOISE)


def _versions(n: str) -> set[str]:
    return set(_VERSION.findall(n))


def _variants(n: str) -> set[str]:
    return set(n.split()) & VARIANTS


def compatible(q: str, c: str) -> bool:
    """Version numbers/sizes and variant words (mini, pro, ...) must agree exactly."""
    return _versions(q) == _versions(c) and _variants(q) == _variants(c)


MATCHER = FuzzyMatcher(normalise=normalise, compatible=compatible, threshold=THRESHOLD)
score = MATCHER.score
best_match = MATCHER.best_match
suggest = MATCHER.suggest
