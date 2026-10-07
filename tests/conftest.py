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
    from modelinfo.domains.ai_models.sources import artificialanalysis, huggingface, lmarena, openrouter

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
