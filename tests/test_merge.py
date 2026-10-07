from modelinfo.core.merge import merge as _merge
from modelinfo.domains.ai_models.domain import DEFAULT_ORDER, PRIORITY
from modelinfo.core import PartialRecord


def merge(records):
    return _merge(records, PRIORITY, DEFAULT_ORDER)


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
