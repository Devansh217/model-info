# Model Info Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A notebook-friendly Python library: `lookup("Claude Opus 5.5")` / `from_csv("models.csv")` return structured specs + benchmarks for public LLMs from OpenRouter, Artificial Analysis, LMArena and Hugging Face, with per-value provenance and explicit gaps.

**Architecture:** One adapter per source turns raw API data into `PartialRecord`s. A fuzzy matcher picks the matching record per source; a merger combines fields by per-field source priority into a `ModelRecord`. All HTTP goes through a disk cache (24h TTL). No LLM.

**Tech Stack:** Python 3.12 (via `uv`), httpx, rapidfuzz, pandas, pyarrow, pytest.

Spec: `docs/superpowers/specs/2026-10-06-model-info-lookup-design.md`

---

## File Structure

```
pyproject.toml
README.md
examples/models.csv
src/modelinfo/__init__.py          # public lookup()/from_csv() + exports
src/modelinfo/models.py            # DEFAULT_FIELDS, PartialRecord, Field, ModelRecord, ModelNotFound
src/modelinfo/matcher.py           # normalise, score, best_match, suggest
src/modelinfo/cache.py             # Cache: disk-cached HTTP GET
src/modelinfo/merge.py             # PRIORITY + merge()
src/modelinfo/api.py               # ModelInfo: lookup(), from_csv()
src/modelinfo/sources/__init__.py
src/modelinfo/sources/base.py      # Fetcher/CatalogSource protocols, SourceUnavailable
src/modelinfo/sources/openrouter.py
src/modelinfo/sources/lmarena.py
src/modelinfo/sources/artificialanalysis.py
src/modelinfo/sources/huggingface.py
tests/conftest.py                  # FakeFetcher + fixture loading
tests/fixtures/openrouter.json
tests/fixtures/artificialanalysis.json
tests/fixtures/hf_llama.json
tests/test_models.py
tests/test_matcher.py
tests/test_cache.py
tests/test_sources.py
tests/test_merge.py
tests/test_api.py
tests/test_live.py                 # @pytest.mark.live, skipped by default
```

All commands run from the repo root.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/modelinfo/__init__.py`, `src/modelinfo/sources/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "modelinfo"
version = "0.1.0"
description = "Look up specs and benchmarks for public LLMs"
requires-python = ">=3.11"
dependencies = ["httpx>=0.27", "rapidfuzz>=3.9", "pandas>=2.2", "pyarrow>=16"]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = ["live: hits real network sources"]
addopts = "-m 'not live'"
```

- [ ] **Step 2: Create empty package files**

`src/modelinfo/__init__.py` and `src/modelinfo/sources/__init__.py` — both empty.

- [ ] **Step 3: Write smoke test** — `tests/test_smoke.py`

```python
def test_package_imports():
    import modelinfo  # noqa: F401
```

- [ ] **Step 4: Install and run**

Run: `uv sync --python 3.12 && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
printf ".venv/\n__pycache__/\n*.egg-info/\n" > .gitignore
git add .gitignore pyproject.toml uv.lock src tests
git commit -m "chore: scaffold modelinfo package"
```

---

### Task 2: Data models

**Files:**
- Create: `src/modelinfo/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test** — `tests/test_models.py`

```python
from modelinfo.models import DEFAULT_FIELDS, Field, ModelNotFound, ModelRecord


def _rec():
    return ModelRecord(
        query="x",
        fields={
            "context_window": Field(1000, "openrouter", "https://o", "t"),
            "arena_elo": Field(1500.0, "lmarena", "https://l", "t"),
        },
    )


def test_gaps_lists_missing_default_fields_in_order():
    gaps = _rec().gaps
    assert gaps == [f for f in DEFAULT_FIELDS if f not in ("context_window", "arena_elo")]


def test_to_dict_returns_values_and_none_for_missing():
    assert _rec().to_dict(["context_window", "swe_bench"]) == {"context_window": 1000, "swe_bench": None}


def test_sources_maps_field_to_source():
    assert _rec().sources == {"context_window": "openrouter", "arena_elo": "lmarena"}


def test_table_has_one_row_per_default_field():
    t = _rec().table()
    assert list(t.columns) == ["field", "value", "source"]
    assert len(t) == len(DEFAULT_FIELDS)
    row = t[t.field == "arena_elo"].iloc[0]
    assert row.value == 1500.0 and row.source == "lmarena"


def test_model_not_found_message_includes_suggestions():
    e = ModelNotFound("foo", ["a", "b"])
    assert e.suggestions == ["a", "b"]
    assert "a, b" in str(e)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'modelinfo.models'`

- [ ] **Step 3: Implement** — `src/modelinfo/models.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class PartialRecord:
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
class ModelRecord:
    query: str
    fields: dict[str, Field]
    matches: dict[str, tuple[str, float]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def gaps(self) -> list[str]:
        return [f for f in DEFAULT_FIELDS if f not in self.fields]

    @property
    def sources(self) -> dict[str, str]:
        return {k: v.source for k, v in self.fields.items()}

    def to_dict(self, fields: list[str] | None = None) -> dict[str, Any]:
        wanted = fields or DEFAULT_FIELDS
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
                for f in DEFAULT_FIELDS
            ]
        )


class ModelNotFound(LookupError):
    def __init__(self, query: str, suggestions: list[str]):
        self.query = query
        self.suggestions = suggestions
        hint = ", ".join(suggestions) or "n/a"
        super().__init__(f"No source matched {query!r}. Did you mean: {hint}?")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_models.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/modelinfo/models.py tests/test_models.py
git commit -m "feat: add ModelRecord and related data models"
```

---

### Task 3: Name matcher

Rules: normalise names (lowercase; strip trailing `(...)`, `"Vendor: "` prefix, `:suffix`, `org/`, dates, effort tags like `high`/`thinking`), then require **identical version numbers** and **identical variant words** (mini, pro, opus, ...), then score with `token_set_ratio` (threshold 85). Ties broken by plain `ratio`; first candidate wins exact ties. (Logic pre-validated against these exact test cases.)

**Files:**
- Create: `src/modelinfo/matcher.py`
- Test: `tests/test_matcher.py`

- [ ] **Step 1: Write the failing test** — `tests/test_matcher.py`

```python
import pytest

from modelinfo.matcher import best_match, normalise, suggest


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Anthropic: Claude Opus 5.5", "claude opus 5.5"),
        ("anthropic/claude-opus-5.5:batch", "claude opus 5.5"),
        ("Anthropic: Claude Opus 5.5 (batch)", "claude opus 5.5"),
        ("claude-opus-5.5-high", "claude opus 5.5"),
        ("gpt-4o-2024-08-06", "gpt 4o"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_best_match_picks_exact_version():
    cands = ["Anthropic: Claude Opus 5", "Anthropic: Claude Opus 5.5", "Anthropic: Claude Sonnet 5.5"]
    assert best_match("Claude Opus 5.5", cands) == (1, 100.0)


def test_partial_query_matches():
    assert best_match("opus 5.5", ["Anthropic: Claude Opus 5.5"]) == (0, 100.0)


def test_version_mismatch_rejected():
    assert best_match("opus 5.5", ["claude-opus-5"]) is None


def test_variant_mismatch_rejected():
    assert best_match("gpt 5", ["gpt-5-mini"]) is None


def test_size_is_a_version_token():
    cands = ["Meta: Llama 3.1 70B Instruct", "Meta: Llama 3.1 8B Instruct"]
    assert best_match("Llama 3.1 8B", cands) == (1, 100.0)


def test_suggest_returns_closest_three():
    cands = ["Anthropic: Claude Opus 5.5", "Google: Gemini 4 Ultra", "Google: Gemini 4 Pro", "Meta: Llama 3.1 8B Instruct"]
    out = suggest("Gemini 9 Ultra", cands)
    assert len(out) == 3
    assert out[0] == "Google: Gemini 4 Ultra"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_matcher.py -q`
Expected: FAIL — `No module named 'modelinfo.matcher'`

- [ ] **Step 3: Implement** — `src/modelinfo/matcher.py`

```python
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

THRESHOLD = 85

NOISE = {"high", "low", "medium", "thinking", "reasoning", "preview", "latest",
         "instruct", "chat", "exp", "experimental"}
VARIANTS = {"mini", "nano", "flash", "lite", "pro", "max", "small", "large", "turbo",
            "haiku", "sonnet", "opus", "plus", "ultra", "air", "fast", "codex", "coder", "vision"}

_VERSION = re.compile(r"\d+(?:\.\d+)*")
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
    return " ".join(t for t in s.split() if t not in NOISE)


def _versions(n: str) -> set[str]:
    return set(_VERSION.findall(n))


def _variants(n: str) -> set[str]:
    return set(n.split()) & VARIANTS


def score(query: str, candidate: str) -> float:
    q, c = normalise(query), normalise(candidate)
    if not q or not c or _versions(q) != _versions(c) or _variants(q) != _variants(c):
        return 0.0
    return float(fuzz.token_set_ratio(q, c))


def best_match(query: str, candidates: list[str], threshold: float = THRESHOLD) -> tuple[int, float] | None:
    nq = normalise(query)
    best = None
    for i, cand in enumerate(candidates):
        s = score(query, cand)
        if s < threshold:
            continue
        key = (s, fuzz.ratio(nq, normalise(cand)))
        if best is None or key > best[0]:
            best = (key, i)
    return (best[1], best[0][0]) if best else None


def suggest(query: str, candidates: list[str], limit: int = 3) -> list[str]:
    uniq = list(dict.fromkeys(candidates))
    hits = process.extract(query, uniq, scorer=fuzz.token_set_ratio, processor=normalise, limit=limit)
    return [c for c, _, _ in hits]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_matcher.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/modelinfo/matcher.py tests/test_matcher.py
git commit -m "feat: add fuzzy model name matcher with version/variant guards"
```

---

### Task 4: Disk cache

**Files:**
- Create: `src/modelinfo/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test** — `tests/test_cache.py`

```python
import httpx
import pytest

from modelinfo.cache import Cache


def _cache(tmp_path, calls, status=200, ttl=3600):
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(status, content=b'{"ok": 1}')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Cache(directory=tmp_path, ttl=ttl, client=client)


def test_second_get_served_from_cache(tmp_path):
    calls = []
    c = _cache(tmp_path, calls)
    body1, t1 = c.get("https://x/a")
    body2, t2 = c.get("https://x/a")
    assert body1 == body2 == b'{"ok": 1}'
    assert t1 == t2
    assert len(calls) == 1


def test_refresh_bypasses_cache(tmp_path):
    calls = []
    c = _cache(tmp_path, calls)
    c.get("https://x/a")
    c.get("https://x/a", refresh=True)
    assert len(calls) == 2


def test_expired_entry_refetched(tmp_path):
    calls = []
    c = _cache(tmp_path, calls, ttl=0)
    c.get("https://x/a")
    c.get("https://x/a")
    assert len(calls) == 2


def test_http_error_raises(tmp_path):
    c = _cache(tmp_path, [], status=500)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("https://x/a")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cache.py -q`
Expected: FAIL — `No module named 'modelinfo.cache'`

- [ ] **Step 3: Implement** — `src/modelinfo/cache.py`

```python
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

DEFAULT_DIR = Path.home() / ".cache" / "modelinfo"
DEFAULT_TTL = 24 * 3600


class Cache:
    def __init__(self, directory: Path | str = DEFAULT_DIR, ttl: float = DEFAULT_TTL,
                 client: httpx.Client | None = None):
        self.directory = Path(directory)
        self.ttl = ttl
        self._client = client or httpx.Client(timeout=60, follow_redirects=True)

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha1(url.encode()).hexdigest()
        return self.directory / f"{key}.bin", self.directory / f"{key}.meta.json"

    def get(self, url: str, headers: dict | None = None, refresh: bool = False) -> tuple[bytes, str]:
        body_path, meta_path = self._paths(url)
        if not refresh and body_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if time.time() - meta["ts"] < self.ttl:
                return body_path.read_bytes(), meta["fetched_at"]
        resp = self._client.get(url, headers=headers or {})
        resp.raise_for_status()
        self.directory.mkdir(parents=True, exist_ok=True)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        body_path.write_bytes(resp.content)
        meta_path.write_text(json.dumps({"url": url, "ts": time.time(), "fetched_at": fetched_at}))
        return resp.content, fetched_at
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cache.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/modelinfo/cache.py tests/test_cache.py
git commit -m "feat: add disk-cached HTTP fetcher"
```

---

### Task 5: Source base, test fixtures, OpenRouter adapter

Fixture numbers are **fictional test data**, not real model specs.

**Files:**
- Create: `src/modelinfo/sources/base.py`, `src/modelinfo/sources/openrouter.py`
- Create (stubs, filled in Tasks 6–8): `src/modelinfo/sources/lmarena.py`, `artificialanalysis.py`, `huggingface.py`
- Create: `tests/conftest.py`, `tests/fixtures/openrouter.json`
- Test: `tests/test_sources.py`

- [ ] **Step 1: Create `src/modelinfo/sources/base.py`**

```python
from __future__ import annotations

from typing import Protocol

from ..models import PartialRecord


class SourceUnavailable(RuntimeError):
    """Source cannot be used (e.g. missing API key)."""


class Fetcher(Protocol):
    def get(self, url: str, headers: dict | None = None, refresh: bool = False) -> tuple[bytes, str]: ...


class CatalogSource(Protocol):
    name: str

    def fetch_all(self, refresh: bool = False) -> list[PartialRecord]: ...


def drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v == v}  # v == v drops NaN
```

- [ ] **Step 2: Create `tests/fixtures/openrouter.json`**

```json
{"data": [
  {"id": "anthropic/claude-opus-5.5", "name": "Anthropic: Claude Opus 5.5", "created": 1788000000,
   "hugging_face_id": null, "context_length": 1000000,
   "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"], "output_modalities": ["text"]},
   "pricing": {"prompt": "0.000005", "completion": "0.000025"},
   "top_provider": {"context_length": 1000000, "max_completion_tokens": 128000},
   "benchmarks": {"artificial_analysis": {"intelligence_index": 72.4}}},
  {"id": "anthropic/claude-opus-5.5:batch", "name": "Anthropic: Claude Opus 5.5 (batch)", "created": 1788000000,
   "context_length": 1000000, "architecture": {}, "pricing": {"prompt": "0.0000025", "completion": "0.0000125"}, "top_provider": {}},
  {"id": "~anthropic/claude-opus-latest", "name": "Anthropic: Claude Opus Latest", "created": 1788000000,
   "context_length": 1000000, "architecture": {}, "pricing": {}, "top_provider": {}},
  {"id": "anthropic/claude-opus-5", "name": "Anthropic: Claude Opus 5", "created": 1780000000,
   "context_length": 500000, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
   "pricing": {"prompt": "0.000015", "completion": "0.000075"}, "top_provider": {"max_completion_tokens": 64000}},
  {"id": "meta-llama/llama-3.1-8b-instruct", "name": "Meta: Llama 3.1 8B Instruct", "created": 1721692800,
   "hugging_face_id": "meta-llama/Llama-3.1-8B-Instruct", "context_length": 131072,
   "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
   "pricing": {"prompt": "0.00000002", "completion": "0.00000005"},
   "top_provider": {"max_completion_tokens": 16384}}
]}
```

- [ ] **Step 3: Create stub modules** (so `conftest.py` imports resolve; replaced in Tasks 6–8)

`src/modelinfo/sources/lmarena.py`:
```python
URL = "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/refs%2Fconvert%2Fparquet/text/latest/0000.parquet"
```
`src/modelinfo/sources/artificialanalysis.py`:
```python
URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
```
`src/modelinfo/sources/huggingface.py`:
```python
API = "https://huggingface.co/api/models/{id}"
```

- [ ] **Step 4: Create `tests/conftest.py`** (also serves the LMArena parquet and the AA/HF fixtures added in later tasks)

```python
import io
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FETCHED_AT = "2026-10-06T00:00:00+00:00"


def lmarena_parquet() -> bytes:
    df = pd.DataFrame(
        [
            {"model_name": "claude-opus-5.5-high", "organization": "anthropic", "license": "Proprietary",
             "rating": 1511.68, "category": "overall"},
            {"model_name": "claude-opus-5.5-high", "organization": "anthropic", "license": "Proprietary",
             "rating": 1530.0, "category": "coding"},
            {"model_name": "llama-3.1-8b-instruct", "organization": "meta", "license": "Llama 3.1 Community",
             "rating": 1176.31, "category": "overall"},
        ]
    )
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class FakeFetcher:
    def __init__(self, responses: dict[str, bytes]):
        self.responses = responses
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, headers=None, refresh=False):
        self.calls.append((url, headers))
        if url not in self.responses:
            raise RuntimeError(f"unexpected url {url}")
        return self.responses[url], FETCHED_AT


@pytest.fixture
def fetcher():
    from modelinfo.sources import artificialanalysis, huggingface, lmarena, openrouter

    responses = {
        openrouter.URL: (FIXTURES / "openrouter.json").read_bytes(),
        lmarena.URL: lmarena_parquet(),
    }
    aa = FIXTURES / "artificialanalysis.json"
    if aa.exists():
        responses[artificialanalysis.URL] = aa.read_bytes()
    hf = FIXTURES / "hf_llama.json"
    if hf.exists():
        responses[huggingface.API.format(id="meta-llama/Llama-3.1-8B-Instruct")] = hf.read_bytes()
    return FakeFetcher(responses)
```

- [ ] **Step 5: Write failing test** — `tests/test_sources.py`

```python
import pytest

from modelinfo.sources.openrouter import OpenRouterSource


def test_openrouter_skips_batch_and_alias_ids(fetcher):
    ids = [r.source_id for r in OpenRouterSource(fetcher).fetch_all()]
    assert ids == ["anthropic/claude-opus-5.5", "anthropic/claude-opus-5", "meta-llama/llama-3.1-8b-instruct"]


def test_openrouter_maps_fields(fetcher):
    rec = OpenRouterSource(fetcher).fetch_all()[0]
    assert rec.source == "openrouter"
    assert rec.name == "Anthropic: Claude Opus 5.5"
    assert rec.url == "https://openrouter.ai/anthropic/claude-opus-5.5"
    assert rec.fields == {
        "name": "Claude Opus 5.5",
        "provider": "Anthropic",
        "context_window": 1000000,
        "max_output_tokens": 128000,
        "input_price_per_mtok": 5.0,
        "output_price_per_mtok": 25.0,
        "modalities": "text,image->text",
        "release_date": "2026-08-29",
        "aa_intelligence_index": 72.4,
    }


def test_openrouter_hf_id_marks_open_weights(fetcher):
    rec = OpenRouterSource(fetcher).fetch_all()[2]
    assert rec.fields["open_weights"] is True
    assert rec.fields["hugging_face_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert rec.fields["input_price_per_mtok"] == 0.02
```

- [ ] **Step 6: Run to verify failure**

Run: `uv run pytest tests/test_sources.py -q`
Expected: FAIL — `No module named 'modelinfo.sources.openrouter'`

- [ ] **Step 7: Implement** — `src/modelinfo/sources/openrouter.py`

```python
from __future__ import annotations

import json
from datetime import datetime, timezone

from ..models import PartialRecord
from .base import Fetcher, drop_none

URL = "https://openrouter.ai/api/v1/models"


class OpenRouterSource:
    name = "openrouter"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def fetch_all(self, refresh: bool = False) -> list[PartialRecord]:
        raw, fetched_at = self.fetcher.get(URL, refresh=refresh)
        out = []
        for m in json.loads(raw)["data"]:
            mid = m["id"]
            if mid.startswith("~") or ":" in mid:
                continue
            out.append(PartialRecord(self.name, mid, m["name"], f"https://openrouter.ai/{mid}",
                                     fetched_at, _fields(m)))
        return out


def _per_mtok(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v * 1_000_000, 4) if v >= 0 else None


def _modalities(arch: dict) -> str | None:
    ins, outs = arch.get("input_modalities") or [], arch.get("output_modalities") or []
    if ins or outs:
        return f"{','.join(ins)}->{','.join(outs)}"
    return arch.get("modality")


def _fields(m: dict) -> dict:
    arch = m.get("architecture") or {}
    top = m.get("top_provider") or {}
    pricing = m.get("pricing") or {}
    aa = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
    name = m.get("name", "")
    provider = name.split(": ", 1)[0] if ": " in name else m["id"].split("/", 1)[0]
    created = m.get("created")
    hf_id = m.get("hugging_face_id") or None
    return drop_none({
        "name": name.split(": ", 1)[-1],
        "provider": provider,
        "context_window": m.get("context_length"),
        "max_output_tokens": top.get("max_completion_tokens"),
        "input_price_per_mtok": _per_mtok(pricing.get("prompt")),
        "output_price_per_mtok": _per_mtok(pricing.get("completion")),
        "modalities": _modalities(arch),
        "release_date": datetime.fromtimestamp(created, tz=timezone.utc).date().isoformat() if created else None,
        "open_weights": True if hf_id else None,
        "hugging_face_id": hf_id,
        "aa_intelligence_index": aa.get("intelligence_index"),
    })
```

- [ ] **Step 8: Run to verify pass**

Run: `uv run pytest tests/test_sources.py -q`
Expected: `3 passed`

- [ ] **Step 9: Commit**

```bash
git add src/modelinfo/sources tests/conftest.py tests/fixtures/openrouter.json tests/test_sources.py
git commit -m "feat: add source base and OpenRouter adapter"
```

---

### Task 6: LMArena adapter

**Files:**
- Modify (replace stub): `src/modelinfo/sources/lmarena.py`
- Test: `tests/test_sources.py` (append)

- [ ] **Step 1: Append failing tests** to `tests/test_sources.py`

```python
from modelinfo.sources.lmarena import LMArenaSource


def test_lmarena_keeps_only_overall_category(fetcher):
    recs = LMArenaSource(fetcher).fetch_all()
    assert [r.name for r in recs] == ["claude-opus-5.5-high", "llama-3.1-8b-instruct"]


def test_lmarena_fields(fetcher):
    opus, llama = LMArenaSource(fetcher).fetch_all()
    assert opus.fields == {"arena_elo": 1511.7, "provider": "anthropic", "license": "Proprietary",
                           "open_weights": False}
    assert llama.fields["open_weights"] is True
    assert opus.url == "https://lmarena.ai/leaderboard/text"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sources.py -q -k lmarena`
Expected: FAIL — `cannot import name 'LMArenaSource'`

- [ ] **Step 3: Implement** — replace `src/modelinfo/sources/lmarena.py`

```python
from __future__ import annotations

import io

import pandas as pd

from ..models import PartialRecord
from .base import Fetcher, drop_none

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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sources.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/modelinfo/sources/lmarena.py tests/test_sources.py
git commit -m "feat: add LMArena leaderboard adapter"
```

---

### Task 7: Artificial Analysis adapter

**Files:**
- Modify (replace stub): `src/modelinfo/sources/artificialanalysis.py`
- Create: `tests/fixtures/artificialanalysis.json`
- Test: `tests/test_sources.py` (append)

- [ ] **Step 1: Create fixture** `tests/fixtures/artificialanalysis.json`

```json
{"status": 200, "data": [
  {"id": "aa-opus-55", "name": "Claude Opus 5.5", "slug": "claude-opus-5-5",
   "model_creator": {"id": "c1", "name": "Anthropic", "slug": "anthropic"},
   "evaluations": {"artificial_analysis_intelligence_index": 73.1, "mmlu_pro": 0.912, "gpqa": 0.874},
   "pricing": {"price_1m_input_tokens": 5.0, "price_1m_output_tokens": 25.0},
   "median_output_tokens_per_second": 61.2, "median_time_to_first_token_seconds": 1.9},
  {"id": "aa-llama", "name": "Llama 3.1 8B Instruct", "slug": "llama-3-1-instruct-8b",
   "model_creator": {"id": "c2", "name": "Meta", "slug": "meta"},
   "evaluations": {"artificial_analysis_intelligence_index": 21.0},
   "pricing": {}, "median_output_tokens_per_second": 0, "median_time_to_first_token_seconds": 0}
]}
```

- [ ] **Step 2: Append failing tests** to `tests/test_sources.py`

```python
from modelinfo.sources.artificialanalysis import ArtificialAnalysisSource
from modelinfo.sources.base import SourceUnavailable


def test_aa_requires_key(fetcher, monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    with pytest.raises(SourceUnavailable):
        ArtificialAnalysisSource(fetcher).fetch_all()


def test_aa_sends_key_and_maps_fields(fetcher):
    opus, llama = ArtificialAnalysisSource(fetcher, api_key="k").fetch_all()
    assert fetcher.calls[-1][1] == {"x-api-key": "k"}
    assert opus.url == "https://artificialanalysis.ai/models/claude-opus-5-5"
    assert opus.fields == {
        "name": "Claude Opus 5.5", "provider": "Anthropic", "aa_intelligence_index": 73.1,
        "mmlu_pro": 0.912, "gpqa": 0.874, "input_price_per_mtok": 5.0, "output_price_per_mtok": 25.0,
        "output_speed_tps": 61.2, "latency_ttft_s": 1.9,
    }
    assert "output_speed_tps" not in llama.fields  # 0 means "not measured"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_sources.py -q -k aa`
Expected: FAIL — `cannot import name 'ArtificialAnalysisSource'`

- [ ] **Step 4: Implement** — replace `src/modelinfo/sources/artificialanalysis.py`

```python
from __future__ import annotations

import json
import os

from ..models import PartialRecord
from .base import Fetcher, SourceUnavailable, drop_none

URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


class ArtificialAnalysisSource:
    name = "artificialanalysis"

    def __init__(self, fetcher: Fetcher, api_key: str | None = None):
        self.fetcher = fetcher
        self.api_key = api_key or os.environ.get("AA_API_KEY")

    def fetch_all(self, refresh: bool = False) -> list[PartialRecord]:
        if not self.api_key:
            raise SourceUnavailable("AA_API_KEY not set (free key: https://artificialanalysis.ai/api-reference)")
        raw, fetched_at = self.fetcher.get(URL, headers={"x-api-key": self.api_key}, refresh=refresh)
        out = []
        for m in json.loads(raw)["data"]:
            ev = m.get("evaluations") or {}
            pr = m.get("pricing") or {}
            swe = ev.get("swe_bench")
            if swe is None:
                swe = ev.get("swe_bench_verified")
            fields = drop_none({
                "name": m.get("name"),
                "provider": (m.get("model_creator") or {}).get("name"),
                "aa_intelligence_index": ev.get("artificial_analysis_intelligence_index"),
                "mmlu_pro": ev.get("mmlu_pro"),
                "gpqa": ev.get("gpqa"),
                "swe_bench": swe,
                "input_price_per_mtok": pr.get("price_1m_input_tokens"),
                "output_price_per_mtok": pr.get("price_1m_output_tokens"),
                "output_speed_tps": m.get("median_output_tokens_per_second") or None,
                "latency_ttft_s": m.get("median_time_to_first_token_seconds") or None,
            })
            url = f"https://artificialanalysis.ai/models/{m.get('slug', '')}"
            out.append(PartialRecord(self.name, m["id"], m["name"], url, fetched_at, fields))
        return out
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_sources.py -q`
Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add src/modelinfo/sources/artificialanalysis.py tests/fixtures/artificialanalysis.json tests/test_sources.py
git commit -m "feat: add Artificial Analysis adapter"
```

---

### Task 8: Hugging Face adapter

Only called with the exact `hugging_face_id` OpenRouter reports — never a free-text search (avoids matching random fine-tunes).

**Files:**
- Modify (replace stub): `src/modelinfo/sources/huggingface.py`
- Create: `tests/fixtures/hf_llama.json`
- Test: `tests/test_sources.py` (append)

- [ ] **Step 1: Create fixture** `tests/fixtures/hf_llama.json`

```json
{"id": "meta-llama/Llama-3.1-8B-Instruct", "cardData": {"license": "llama3.1"}}
```

- [ ] **Step 2: Append failing test** to `tests/test_sources.py`

```python
from modelinfo.sources.huggingface import HuggingFaceSource


def test_hf_fetch_one(fetcher):
    rec = HuggingFaceSource(fetcher).fetch_one("meta-llama/Llama-3.1-8B-Instruct")
    assert rec.source == "huggingface"
    assert rec.url == "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    assert rec.fields == {"open_weights": True, "license": "llama3.1"}
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_sources.py -q -k hf`
Expected: FAIL — `cannot import name 'HuggingFaceSource'`

- [ ] **Step 4: Implement** — replace `src/modelinfo/sources/huggingface.py`

```python
from __future__ import annotations

import json

from ..models import PartialRecord
from .base import Fetcher, drop_none

API = "https://huggingface.co/api/models/{id}"


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def fetch_one(self, repo_id: str, refresh: bool = False) -> PartialRecord:
        raw, fetched_at = self.fetcher.get(API.format(id=repo_id), refresh=refresh)
        m = json.loads(raw)
        card = m.get("cardData") or {}
        lic = card.get("license")
        if lic == "other":
            lic = card.get("license_name") or lic
        rid = m.get("id", repo_id)
        return PartialRecord(self.name, rid, rid, f"https://huggingface.co/{rid}", fetched_at,
                             drop_none({"open_weights": True, "license": lic}))
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_sources.py -q`
Expected: `8 passed`

- [ ] **Step 6: Commit**

```bash
git add src/modelinfo/sources/huggingface.py tests/fixtures/hf_llama.json tests/test_sources.py
git commit -m "feat: add Hugging Face adapter"
```

---

### Task 9: Merge by field priority

**Files:**
- Create: `src/modelinfo/merge.py`
- Test: `tests/test_merge.py`

- [ ] **Step 1: Write failing test** — `tests/test_merge.py`

```python
from modelinfo.merge import merge
from modelinfo.models import PartialRecord


def _p(source, **fields):
    return PartialRecord(source, "id", "n", f"https://{source}", "t", fields)


def test_priority_order_per_field():
    out = merge([
        _p("artificialanalysis", input_price_per_mtok=4.0, aa_intelligence_index=73.1),
        _p("openrouter", input_price_per_mtok=5.0, aa_intelligence_index=72.4),
    ])
    assert out["input_price_per_mtok"].value == 5.0
    assert out["input_price_per_mtok"].source == "openrouter"
    assert out["aa_intelligence_index"].value == 73.1
    assert out["aa_intelligence_index"].source == "artificialanalysis"


def test_falls_back_to_lower_priority_source():
    out = merge([_p("lmarena", open_weights=False, license="Proprietary"), _p("openrouter", name="X")])
    assert out["open_weights"].value is False
    assert out["open_weights"].source == "lmarena"
    assert out["license"].url == "https://lmarena"


def test_huggingface_wins_license():
    out = merge([_p("lmarena", license="Llama"), _p("huggingface", license="llama3.1")])
    assert out["license"].value == "llama3.1"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_merge.py -q`
Expected: FAIL — `No module named 'modelinfo.merge'`

- [ ] **Step 3: Implement** — `src/modelinfo/merge.py`

```python
from __future__ import annotations

from .models import Field, PartialRecord

OR, AA, LM, HF = "openrouter", "artificialanalysis", "lmarena", "huggingface"
DEFAULT_ORDER = [OR, AA, LM, HF]

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


def merge(records: list[PartialRecord]) -> dict[str, Field]:
    by_source = {r.source: r for r in records}
    names = {k for r in records for k in r.fields}
    out: dict[str, Field] = {}
    for f in sorted(names):
        for src in PRIORITY.get(f, DEFAULT_ORDER):
            r = by_source.get(src)
            if r is not None and f in r.fields:
                out[f] = Field(r.fields[f], r.source, r.url, r.fetched_at)
                break
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_merge.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/modelinfo/merge.py tests/test_merge.py
git commit -m "feat: merge partial records by per-field source priority"
```

---

### Task 10: Public API — `lookup()` and `from_csv()`

**Files:**
- Create: `src/modelinfo/api.py`
- Modify: `src/modelinfo/__init__.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing test** — `tests/test_api.py`

```python
import pytest

from modelinfo.api import ModelInfo
from modelinfo.models import ModelNotFound


@pytest.fixture
def mi(fetcher):
    return ModelInfo(cache=fetcher, aa_api_key="k")


def test_lookup_merges_all_sources(mi):
    rec = mi.lookup("Claude Opus 5.5")
    assert rec.fields["context_window"].value == 1000000
    assert rec.fields["context_window"].source == "openrouter"
    assert rec.fields["aa_intelligence_index"].value == 73.1
    assert rec.fields["arena_elo"].value == 1511.7
    assert rec.fields["open_weights"].value is False
    assert rec.fields["mmlu_pro"].source == "artificialanalysis"
    assert rec.gaps == ["swe_bench"]
    assert rec.matches["openrouter"] == ("Anthropic: Claude Opus 5.5", 100.0)
    assert rec.errors == {}


def test_lookup_without_aa_key_warns_and_falls_back(fetcher, monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    with pytest.warns(UserWarning, match="artificialanalysis"):
        rec = ModelInfo(cache=fetcher).lookup("Claude Opus 5.5")
    assert "artificialanalysis" in rec.errors
    assert rec.fields["aa_intelligence_index"].value == 72.4
    assert rec.fields["aa_intelligence_index"].source == "openrouter"


def test_lookup_open_model_uses_huggingface(mi):
    rec = mi.lookup("Llama 3.1 8B Instruct")
    assert rec.fields["license"].value == "llama3.1"
    assert rec.fields["license"].source == "huggingface"
    assert rec.fields["open_weights"].value is True
    assert "huggingface" in rec.matches


def test_lookup_unknown_model_raises_with_suggestions(mi):
    with pytest.raises(ModelNotFound) as e:
        mi.lookup("Gemini 9 Ultra")
    assert 0 < len(e.value.suggestions) <= 3


def test_from_csv_selected_fields(mi, tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("model,context_window,arena_elo,swe_bench\nClaude Opus 5.5\nLlama 3.1 8B\nGemini 9 Ultra\n")
    df = mi.from_csv(p)
    assert list(df.columns) == ["model", "context_window", "arena_elo", "swe_bench",
                                "matched_as", "match_score", "gaps"]
    opus, llama, missing = df.to_dict("records")
    assert opus["context_window"] == 1000000 and opus["gaps"] == "swe_bench"
    assert llama["arena_elo"] == 1176.3
    assert missing["gaps"].startswith("NOT FOUND")


def test_from_csv_defaults_to_all_fields(mi, tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("model\nClaude Opus 5.5\n")
    df = mi.from_csv(p)
    assert "latency_ttft_s" in df.columns and "license" in df.columns


def test_from_csv_rejects_unknown_columns(mi, tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("model,bogus\nx,\n")
    with pytest.raises(ValueError, match="bogus"):
        mi.from_csv(p)


def test_from_csv_requires_model_column(mi, tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("name\nx\n")
    with pytest.raises(ValueError, match="model"):
        mi.from_csv(p)


def test_sources_loaded_once_per_csv(mi, fetcher, tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("model\nClaude Opus 5.5\nClaude Opus 5\n")
    mi.from_csv(p)
    urls = [u for u, _ in fetcher.calls]
    assert urls.count("https://openrouter.ai/api/v1/models") == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_api.py -q`
Expected: FAIL — `No module named 'modelinfo.api'`

- [ ] **Step 3: Implement** — `src/modelinfo/api.py`

```python
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .cache import Cache
from .matcher import best_match, suggest
from .merge import merge
from .models import DEFAULT_FIELDS, ModelNotFound, ModelRecord, PartialRecord
from .sources.artificialanalysis import ArtificialAnalysisSource
from .sources.base import CatalogSource, Fetcher
from .sources.huggingface import HuggingFaceSource
from .sources.lmarena import LMArenaSource
from .sources.openrouter import OpenRouterSource


class ModelInfo:
    def __init__(self, cache: Fetcher | None = None, aa_api_key: str | None = None):
        cache = cache or Cache()
        self.catalogs: list[CatalogSource] = [
            OpenRouterSource(cache),
            ArtificialAnalysisSource(cache, aa_api_key),
            LMArenaSource(cache),
        ]
        self.huggingface = HuggingFaceSource(cache)

    def _load(self, refresh: bool) -> tuple[dict[str, list[PartialRecord]], dict[str, str]]:
        data, errors = {}, {}
        for src in self.catalogs:
            try:
                data[src.name] = src.fetch_all(refresh=refresh)
            except Exception as e:  # one bad source must not break the lookup
                errors[src.name] = f"{type(e).__name__}: {e}"
                warnings.warn(f"modelinfo: skipping {src.name}: {errors[src.name]}", UserWarning, stacklevel=3)
        return data, errors

    def _lookup(self, query: str, data, errors, refresh: bool) -> ModelRecord:
        errors = dict(errors)
        matched: list[PartialRecord] = []
        matches: dict[str, tuple[str, float]] = {}
        for src_name, recs in data.items():
            hit = best_match(query, [r.name for r in recs])
            if hit:
                idx, score = hit
                matched.append(recs[idx])
                matches[src_name] = (recs[idx].name, score)
        if not matched:
            names = [r.name for recs in data.values() for r in recs]
            raise ModelNotFound(query, suggest(query, names))

        hf_id = next((r.fields["hugging_face_id"] for r in matched if r.fields.get("hugging_face_id")), None)
        if hf_id:
            try:
                hf = self.huggingface.fetch_one(hf_id, refresh=refresh)
                matched.append(hf)
                matches["huggingface"] = (hf.name, 100.0)
            except Exception as e:
                errors["huggingface"] = f"{type(e).__name__}: {e}"
        return ModelRecord(query=query, fields=merge(matched), matches=matches, errors=errors)

    def lookup(self, query: str, refresh: bool = False) -> ModelRecord:
        data, errors = self._load(refresh)
        return self._lookup(query, data, errors, refresh)

    def from_csv(self, path: str | Path, refresh: bool = False) -> pd.DataFrame:
        src = pd.read_csv(path)
        if "model" not in src.columns:
            raise ValueError("CSV must have a 'model' column")
        requested = [c for c in src.columns if c != "model"]
        unknown = [c for c in requested if c not in DEFAULT_FIELDS]
        if unknown:
            raise ValueError(f"Unknown field columns: {unknown}. Known fields: {DEFAULT_FIELDS}")
        fields = requested or DEFAULT_FIELDS

        data, errors = self._load(refresh)
        rows = []
        for query in src["model"].astype(str):
            row: dict = {"model": query}
            try:
                rec = self._lookup(query, data, errors, refresh)
                row.update(rec.to_dict(fields))
                best = rec.matches.get("openrouter") or max(rec.matches.values(), key=lambda m: m[1])
                row["matched_as"], row["match_score"] = best
                row["gaps"] = ", ".join(f for f in fields if row[f] is None)
            except ModelNotFound as e:
                row.update({f: None for f in fields})
                row["matched_as"], row["match_score"] = None, None
                row["gaps"] = "NOT FOUND; suggestions: " + ", ".join(e.suggestions)
            rows.append(row)
        return pd.DataFrame(rows, columns=["model", *fields, "matched_as", "match_score", "gaps"])
```

- [ ] **Step 4: Replace `src/modelinfo/__init__.py`**

```python
from __future__ import annotations

from pathlib import Path

from .api import ModelInfo
from .models import DEFAULT_FIELDS, Field, ModelNotFound, ModelRecord

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
```

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: `41 passed`

- [ ] **Step 6: Commit**

```bash
git add src/modelinfo/api.py src/modelinfo/__init__.py tests/test_api.py
git commit -m "feat: add lookup() and from_csv() public API"
```

---

### Task 11: Live smoke test, example CSV, README

**Files:**
- Create: `tests/test_live.py`, `examples/models.csv`, `README.md`
- Delete: `tests/test_smoke.py` (superseded)

- [ ] **Step 1: Create `tests/test_live.py`**

```python
import pytest

from modelinfo import ModelInfo
from modelinfo.cache import Cache


@pytest.mark.live
def test_live_lookup(tmp_path):
    rec = ModelInfo(cache=Cache(directory=tmp_path)).lookup("Claude Opus 5.5")
    assert rec.fields["context_window"].value > 0
    assert "openrouter" in rec.matches
    assert "lmarena" in rec.matches
```

- [ ] **Step 2: Run the live test**

Run: `uv run pytest -m live -q`
Expected: `1 passed` (a `UserWarning` for `artificialanalysis` is expected if `AA_API_KEY` is unset). If it fails, find which source broke (real API shape may have changed), fix the adapter, and add a fixture-based regression test.

- [ ] **Step 3: Create `examples/models.csv`**

```csv
model,context_window,input_price_per_mtok,output_price_per_mtok,arena_elo,aa_intelligence_index,gpqa
Claude Opus 5.5
GPT-5
Gemini 2.5 Pro
Llama 3.1 8B Instruct
```

- [ ] **Step 4: Create `README.md`**

````markdown
# modelinfo

Look up specs and benchmark numbers for public LLMs from structured sources
(OpenRouter, Artificial Analysis, LMArena, Hugging Face). Every value records its source; missing values are reported as gaps.

## Install
```bash
uv sync
export AA_API_KEY=...   # optional, free key: https://artificialanalysis.ai/api-reference
```

## Use (notebook)
```python
from modelinfo import lookup, from_csv

rec = lookup("Claude Opus 5.5")
rec.table()        # field | value | source
rec.gaps           # fields no source had
rec.matches        # which record each source matched

df = from_csv("examples/models.csv")
```

CSV: a `model` column plus optional columns naming the fields you want (none = all fields).
Fields: name, provider, context_window, max_output_tokens, input_price_per_mtok, output_price_per_mtok,
modalities, release_date, open_weights, license, arena_elo, mmlu_pro, gpqa, swe_bench,
aa_intelligence_index, output_speed_tps, latency_ttft_s.

Responses are cached in `~/.cache/modelinfo/` for 24h; pass `refresh=True` to bypass.

Notes: `release_date` is when the model was listed on OpenRouter; `swe_bench` is usually empty because the free AA API does not expose it.

## Test
```bash
uv run pytest            # offline, fixtures only
uv run pytest -m live    # hits real APIs
```
````

- [ ] **Step 5: Remove the scaffold smoke test and run everything**

Run: `git rm -q tests/test_smoke.py && uv run pytest -q`
Expected: `40 passed, 1 deselected`

- [ ] **Step 6: Commit**

```bash
git add tests/test_live.py examples/models.csv README.md
git commit -m "docs: add README, example CSV and live smoke test"
```
