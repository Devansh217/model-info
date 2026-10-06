# Model Info Lookup — Design

## Goal
Python library (notebook-friendly) that, given a model name or a CSV of models, returns structured
specs and benchmark numbers for any publicly available LLM, using only structured public sources.
No LLM is used. Missing values stay empty and are reported.

Out of scope (separate sub-project): LLM news crawler.

## Usage
```python
from modelinfo import lookup, from_csv
rec = lookup("Claude Opus 5.5")      # -> ModelRecord
rec.to_dict(); rec.gaps; rec.sources
df = from_csv("models.csv")          # -> pandas.DataFrame
```

### CSV input
- Required column `model` (free-text name).
- Optional extra columns: each column name is a requested field (must be a known field name);
  if no extra columns are given, all default fields are returned.
- Output DataFrame: one row per input model, one column per field, plus `matched_as`,
  `match_score`, `gaps`.

## Default fields
name, provider, context_window, max_output_tokens, input_price_per_mtok, output_price_per_mtok,
modalities, release_date, open_weights, license, arena_elo, mmlu_pro, gpqa, swe_bench,
aa_intelligence_index, output_speed_tps, latency_ttft_s

## Sources (adapters)
| Adapter | Access | Fields |
|---|---|---|
| OpenRouter `/api/v1/models` | public, no key | name, provider, context_window, max_output_tokens, prices, modalities, release_date (date added to OpenRouter), aa_intelligence_index (fallback), hugging_face_id |
| Hugging Face Hub API `/api/models/{id}` | public, called only for the `hugging_face_id` OpenRouter reports (avoids matching random fine-tunes) | open_weights, license |
| LMArena `lmarena-ai/leaderboard-dataset` parquet (`text/latest`, category `overall`) | public | arena_elo, license, open_weights (license != Proprietary) |
| Artificial Analysis API | free API key `AA_API_KEY`, optional | aa_intelligence_index, mmlu_pro, gpqa, swe_bench (if exposed), output_speed_tps, latency_ttft_s |

If an optional source is unavailable (no key, HTTP error) it is skipped with a warning; lookup still succeeds.

## Components
- `modelinfo/sources/base.py` — `CatalogSource` protocol: `name`, `fetch_all(refresh) -> list[PartialRecord]`;
  Hugging Face uses `fetch_one(repo_id)` instead.
- `modelinfo/sources/{openrouter,huggingface,lmarena,artificialanalysis}.py` — one adapter each.
- `modelinfo/matcher.py` — normalise names (lowercase, strip vendor prefix, unify `-`/`.`/spaces,
  version tokens, drop dates and effort tags like -high/-thinking) and fuzzy-match with rapidfuzz token_set_ratio;
  version numbers and variant words (mini, pro, flash, sonnet, opus, ...) must match exactly; threshold 85.
- `modelinfo/merge.py` — per-field source priority; each value stored as `Field(value, source, url, fetched_at)`.
- `modelinfo/cache.py` — JSON cache in `~/.cache/modelinfo/`, 24h TTL, `refresh=True` to bypass.
- `modelinfo/api.py` — `lookup()`, `from_csv()`, `ModelRecord` dataclass.

## Field priority
- context_window / max_output / prices / modalities: OpenRouter > HF
- license / open_weights: HF > OpenRouter > LMArena
- benchmarks & speed: Artificial Analysis (aa_intelligence_index falls back to OpenRouter); arena_elo: LMArena
- Note: the AA free API does not currently expose SWE-bench, so `swe_bench` will usually be a gap.

## Error handling
- No match in any source → `ModelNotFound` with top-3 closest suggestions.
- Low-confidence match (score < threshold) in a source → that source ignored for the model.
- Network errors per source → warning, source skipped, recorded in `record.errors`.

## Testing
- pytest with recorded JSON fixtures per source (no network in tests).
- Unit tests: normalisation, fuzzy matching edge cases (e.g. "opus 5.5" vs "opus 5"), merge priority,
  CSV field selection, gap reporting, cache TTL.
- One optional `@pytest.mark.live` smoke test hitting real sources.

## Stack
Python 3.11+, httpx, rapidfuzz, pandas, pytest.
