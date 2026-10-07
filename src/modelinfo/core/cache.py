from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

DEFAULT_DIR = Path.home() / ".cache" / "modelinfo"
DEFAULT_TTL = 24 * 3600
DEFAULT_USER_AGENT = "modelinfo-scraper/0.1 (+https://github.com/)"


class Cache:
    """Caching HTTP fetcher with per-host throttling. Shared by every domain; point `directory`
    at a per-domain folder to keep caches separate."""

    def __init__(self, directory: Path | str = DEFAULT_DIR, ttl: float = DEFAULT_TTL,
                 client: httpx.Client | None = None, min_interval: float = 0.0,
                 default_headers: dict | None = None):
        self.directory = Path(directory)
        self.ttl = ttl
        self.min_interval = min_interval
        self.default_headers = {"User-Agent": DEFAULT_USER_AGENT, **(default_headers or {})}
        self._client = client or httpx.Client(timeout=60, follow_redirects=True)
        self._last_hit: dict[str, float] = {}

    def _paths(self, key_src: str) -> tuple[Path, Path]:
        key = hashlib.sha1(key_src.encode()).hexdigest()
        return self.directory / f"{key}.bin", self.directory / f"{key}.meta.json"

    def _throttle(self, url: str) -> None:
        if self.min_interval <= 0:
            return
        host = urlsplit(url).netloc
        wait = self._last_hit.get(host, 0.0) + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def get(self, url: str, headers: dict | None = None, refresh: bool = False) -> tuple[bytes, str]:
        return self.request("GET", url, headers=headers, refresh=refresh)

    def request(self, method: str, url: str, *, headers: dict | None = None, params: dict | None = None,
                data: dict | None = None, json: object = None, refresh: bool = False) -> tuple[bytes, str]:
        method = method.upper()
        extras = (params, data, json)
        # Plain GETs keep the URL as cache key; anything else includes method + payload.
        key_src = url if method == "GET" and not any(extras) else _dumps([method, url, *extras])
        body_path, meta_path = self._paths(key_src)
        if not refresh and body_path.exists() and meta_path.exists():
            meta = _loads(meta_path.read_text())
            if time.time() - meta["ts"] < self.ttl:
                return body_path.read_bytes(), meta["fetched_at"]
        self._throttle(url)
        resp = self._client.request(method, url, headers={**self.default_headers, **(headers or {})},
                                    params=params, data=data, json=json)
        resp.raise_for_status()
        self.directory.mkdir(parents=True, exist_ok=True)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        body_path.write_bytes(resp.content)
        meta_path.write_text(_dumps({"url": url, "method": method, "ts": time.time(), "fetched_at": fetched_at}))
        return resp.content, fetched_at


def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _loads(s: str):
    return json.loads(s)
