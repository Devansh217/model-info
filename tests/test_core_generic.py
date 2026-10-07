"""Proves the core is domain-agnostic: a toy 'companies' domain built only from core pieces."""

import re

import httpx
import pytest

from modelinfo.core import Cache, Domain, Engine, Enricher, FuzzyMatcher, PartialRecord, drop_none, repeated

T = "2026-10-07T00:00:00+00:00"

REGISTRY_HTML = """
<table>
<tr><td class="name">Acme Widgets Private Limited</td><td class="cin">U12345MH2010PTC000001</td>
    <td class="status">Active</td><td class="web">https://acme.example</td></tr>
<tr><td class="name">Acme Gadgets Limited</td><td class="cin">L99999DL1999PLC000002</td>
    <td class="status">Strike Off</td><td class="web"></td></tr>
</table>
"""

ROW = re.compile(r'class="name">(.*?)<.*?class="cin">(.*?)<.*?class="status">(.*?)<.*?class="web">(.*?)<', re.S)


class RegistrySearch:
    """SearchSource that POSTs a search form and scrapes the HTML result table."""

    name = "registry"

    def __init__(self, fetcher):
        self.fetcher = fetcher

    def search(self, query, refresh=False):
        raw, at = self.fetcher.request("POST", "https://registry.example/search", data={"q": query}, refresh=refresh)
        return [PartialRecord(self.name, cin, name, f"https://registry.example/{cin}", at,
                              drop_none({"name": name, "cin": cin, "status": status, "website": web or None}))
                for name, cin, status, web in ROW.findall(raw.decode())]


class Financials:
    """CatalogSource with a bulk JSON dump."""

    name = "financials"

    def fetch_all(self, refresh=False):
        return [PartialRecord(self.name, "U12345MH2010PTC000001", "ACME WIDGETS PVT LTD", "https://fin.example", T,
                              {"revenue": 120.5, "ebitda": 30.1, "pat": 12.0})]


class WebsiteContacts:
    """DetailSource keyed on the website found by the registry."""

    name = "website"

    def fetch_one(self, key, refresh=False):
        return PartialRecord(self.name, key, key, key, T, {"directors": [
            {"name": "A. Shah", "phone": "+91 1", "email": "a@acme.example"},
            {"name": "B. Rao", "phone": None, "email": "b@acme.example"},
        ]})


def _normalise(s):
    s = re.sub(r"\b(private|pvt|limited|ltd)\b", " ", s.lower())
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


COMPANIES = Domain(
    name="companies",
    key_column="company",
    fields=["name", "cin", "status", "revenue", "ebitda", "pat", "website", "directors"],
    priority={"revenue": ["financials"], "directors": ["website"]},
    default_order=["registry", "financials", "website"],
    matcher=FuzzyMatcher(normalise=_normalise, compatible=lambda q, c: {"widgets", "gadgets"} & set(q.split())
                         <= set(c.split())),
    flatten={"directors": repeated(3, {"name": "Director {i} Name", "phone": "Director {i} Phone",
                                       "email": "Director {i} Email"})},
)


@pytest.fixture
def engine(tmp_path):
    def handler(request):
        assert request.method == "POST"
        return httpx.Response(200, content=REGISTRY_HTML.encode())

    cache = Cache(directory=tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))
    return Engine(COMPANIES, catalogs=[Financials()], searchers=[RegistrySearch(cache)],
                  enrichers=[Enricher("website", WebsiteContacts())])


def test_generic_engine_scrapes_and_merges_non_ai_domain(engine):
    rec = engine.lookup("Acme Widgets Ltd")
    assert rec.fields["cin"].value == "U12345MH2010PTC000001"
    assert rec.fields["revenue"].source == "financials"
    assert rec.fields["directors"].source == "website"
    assert rec.gaps == []


def test_generic_csv_flattens_repeated_groups(engine, tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("company,cin,revenue,directors\nAcme Widgets\n")
    df = engine.from_csv(p)
    assert list(df.columns) == ["company", "cin", "revenue",
                                "Director 1 Name", "Director 1 Phone", "Director 1 Email",
                                "Director 2 Name", "Director 2 Phone", "Director 2 Email",
                                "Director 3 Name", "Director 3 Phone", "Director 3 Email",
                                "matched_as", "match_score", "gaps"]
    row = df.iloc[0]
    assert row["Director 2 Email"] == "b@acme.example" and row["Director 3 Name"] is None


def test_cache_keys_post_payload_separately(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.content)
        return httpx.Response(200, content=request.content)

    c = Cache(directory=tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))
    a, _ = c.request("POST", "https://x/s", data={"q": "a"})
    b, _ = c.request("POST", "https://x/s", data={"q": "b"})
    c.request("POST", "https://x/s", data={"q": "a"})
    assert a != b and len(calls) == 2


def test_cache_sends_user_agent_and_throttles(tmp_path, monkeypatch):
    seen, sleeps = [], []
    monkeypatch.setattr("modelinfo.core.cache.time.sleep", sleeps.append)

    def handler(request):
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, content=b"x")

    c = Cache(directory=tmp_path, min_interval=5, default_headers={"User-Agent": "me"},
              client=httpx.Client(transport=httpx.MockTransport(handler)))
    c.get("https://h/a")
    c.get("https://h/b")
    assert seen == ["me", "me"] and len(sleeps) == 1 and sleeps[0] > 0
