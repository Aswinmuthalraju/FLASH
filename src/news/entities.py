"""Entity linking: ticker <-> company names / aliases (dictionary, regex word-boundary match)."""
from __future__ import annotations

import re

from src.data.company_universe import UNIVERSE
from src.news.event_rules import MARKET_WIDE


class EntityLinker:
    def __init__(self, universe: dict[str, list[str]] | None = None, market_wide_relevance: float = 0.3):
        self.universe = universe or UNIVERSE
        self.mw = market_wide_relevance
        self._pats = []   # (symbol, regex, relevance)
        for sym, names in self.universe.items():
            for i, name in enumerate(names):
                rel = 1.0 if (i == 0 or name == sym) else 0.8
                flags = 0 if (name.isupper() and len(name) <= 10) else re.I   # tickers are case-sensitive
                self._pats.append((sym, re.compile(r"\b" + re.escape(name) + r"\b", flags), rel))

    def link(self, headline: str, event_type: str, symbols: list[str] | None = None) -> dict[str, float]:
        """{symbol: relevance}. Market-wide events with no named company apply to every symbol (low relevance)."""
        found: dict[str, float] = {}
        for sym, pat, rel in self._pats:
            if pat.search(headline):
                found[sym] = max(found.get(sym, 0.0), rel)
        if symbols is not None:
            found = {s: r for s, r in found.items() if s in symbols}
        if not found and event_type in MARKET_WIDE:
            return {s: self.mw for s in (symbols if symbols is not None else self.universe)}
        return found
