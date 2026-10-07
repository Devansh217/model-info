from modelinfo import DEFAULT_FIELDS, Field, ModelNotFound, ModelRecord


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
