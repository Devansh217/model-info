import pytest

from modelinfo.domains.ai_models.sources.openrouter import OpenRouterSource


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

from modelinfo.domains.ai_models.sources.lmarena import LMArenaSource


def test_lmarena_keeps_only_overall_category(fetcher):
    recs = LMArenaSource(fetcher).fetch_all()
    assert [r.name for r in recs] == ["claude-opus-5.5-high", "llama-3.1-8b-instruct"]


def test_lmarena_fields(fetcher):
    opus, llama = LMArenaSource(fetcher).fetch_all()
    assert opus.fields == {"arena_elo": 1511.7, "provider": "anthropic", "license": "Proprietary",
                           "open_weights": False}
    assert llama.fields["open_weights"] is True
    assert opus.url == "https://lmarena.ai/leaderboard/text"

from modelinfo.domains.ai_models.sources.artificialanalysis import ArtificialAnalysisSource
from modelinfo.core.sources import SourceUnavailable


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

from modelinfo.domains.ai_models.sources.huggingface import HuggingFaceSource


def test_hf_fetch_one(fetcher):
    rec = HuggingFaceSource(fetcher).fetch_one("meta-llama/Llama-3.1-8B-Instruct")
    assert rec.source == "huggingface"
    assert rec.url == "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    assert rec.fields == {"open_weights": True, "license": "llama3.1"}
