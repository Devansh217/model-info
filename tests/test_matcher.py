import pytest

from modelinfo.domains.ai_models.matcher import best_match, normalise, suggest


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


def test_hyphenated_claude_minor_version_matches():
    assert best_match("Claude Opus 4.1", ["claude-opus-4-1-20250805"]) == (0, 100.0)


def test_hyphenated_claude_3_5_version_matches():
    assert best_match("Claude 3.5 Sonnet", ["claude-3-5-sonnet-20241022"]) == (0, 100.0)


def test_letter_suffixed_version_does_not_match_unsuffixed_query():
    assert best_match("GPT-4", ["gpt-4o-2024-05-13"]) is None


def test_letter_suffixed_version_matches_same_suffix():
    assert best_match("GPT-4o", ["gpt-4o-2024-05-13"]) == (0, 100.0)


def test_size_suffix_prevents_digit_group_merge():
    assert normalise("llama-3-8b") == "llama 3 8b"


def test_exact_major_version_prefers_non_minor_candidate():
    cands = ["claude-opus-4-1-20250805", "claude-opus-4-20250514"]
    assert best_match("Claude Opus 4", cands) == (1, 100.0)


def test_suggest_returns_closest_three():
    cands = ["Anthropic: Claude Opus 5.5", "Google: Gemini 4 Ultra", "Google: Gemini 4 Pro", "Meta: Llama 3.1 8B Instruct"]
    out = suggest("Gemini 9 Ultra", cands)
    assert len(out) == 3
    assert out[0] == "Google: Gemini 4 Ultra"
