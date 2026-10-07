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

## Architecture: generic core + pluggable domains

```
src/modelinfo/core/              # domain-agnostic, never edited per domain
  cache.py     Cache: HTTP GET/POST + 24h disk cache, User-Agent, per-host throttle (min_interval)
  sources.py   CatalogSource (bulk list) | SearchSource (per-query search) | DetailSource (fetch by key)
  matching.py  FuzzyMatcher(normalise=..., compatible=...)  — or any object with best_match/suggest
  domain.py    Domain (fields, key column, per-field source priority, matcher, CSV flatteners), Enricher
  engine.py    Engine: catalogs + searchers -> match -> enrichers -> merge by priority -> Record / DataFrame
src/modelinfo/domains/ai_models/ # the LLM domain (OpenRouter, AA, LMArena, Hugging Face)
```

### Adding a new domain (e.g. company records)
1. Create `domains/<name>/sources/*.py`. Each source fetches with `fetcher.get(...)` or
   `fetcher.request("POST", url, data=...)`. It parses the bytes however it needs to (JSON, HTML, parquet)
   and returns `PartialRecord`s whose `fields` dict uses your field names.
2. Declare a `Domain`: the field list, the CSV key column (e.g. `company`), `priority` (which source wins
   for each field) and a matcher (e.g. strip "pvt/ltd"). Add `flatten={"directors": repeated(4, {...})}`
   to spread nested lists into numbered CSV columns.
3. Wire it up: `Engine(domain, catalogs=[...], searchers=[...], enrichers=[Enricher("website", ...)])`.

`tests/test_core_generic.py` is a complete worked example: a toy companies domain that scrapes an HTML
search form and fetches directors through an enricher.
Respect each site's terms of service and robots.txt. Use `Cache(min_interval=...)` to limit request rate.

## Test
```bash
uv run pytest            # offline, fixtures only
uv run pytest -m live    # hits real APIs
```
