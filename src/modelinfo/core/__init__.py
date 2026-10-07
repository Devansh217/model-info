"""Domain-agnostic scraping/lookup core. See README "Adding a new domain"."""

from .cache import Cache
from .domain import Domain, Enricher, Flattener, repeated
from .engine import Engine
from .matching import FuzzyMatcher, Matcher
from .merge import merge
from .records import Field, NotFound, PartialRecord, Record
from .sources import (CatalogSource, DetailSource, Fetcher, RequestFetcher, SearchSource,
                      SourceUnavailable, drop_none)

__all__ = ["Cache", "Domain", "Enricher", "Flattener", "repeated", "Engine", "FuzzyMatcher", "Matcher", "merge", "Field",
           "NotFound", "PartialRecord", "Record", "CatalogSource", "DetailSource", "Fetcher", "RequestFetcher", "SearchSource",
           "SourceUnavailable", "drop_none"]
