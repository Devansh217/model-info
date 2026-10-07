import httpx
import pytest

from modelinfo.core.cache import Cache


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
