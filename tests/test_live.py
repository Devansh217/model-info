import pytest

from modelinfo import ModelInfo
from modelinfo.core.cache import Cache


@pytest.mark.live
def test_live_lookup(tmp_path):
    rec = ModelInfo(cache=Cache(directory=tmp_path)).lookup("Claude Opus 5.5")
    assert rec.fields["context_window"].value > 0
    assert "openrouter" in rec.matches
    assert "lmarena" in rec.matches
