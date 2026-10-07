import pytest

from modelinfo import ModelInfo
from modelinfo import ModelNotFound


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
