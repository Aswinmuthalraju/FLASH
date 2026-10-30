"""Event type: keyword rules over a fixed taxonomy (no model)."""
from __future__ import annotations

import re

EVENT_TYPES = ["Earnings", "M&A", "Regulation", "RBI/Rates", "Geopolitical", "Legal/Fraud", "Management", "Other"]
EVENT_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
MARKET_WIDE = {"RBI/Rates", "Geopolitical"}

_KEYWORDS = {
    "Earnings": ["earnings", "profit", "net profit", "quarterly", "q1", "q2", "q3", "q4", "results", "revenue",
                 "ebitda", "margin", "margins", "eps", "guidance", "estimates"],
    "M&A": ["acquire", "acquisition", "merger", "takeover", "stake", "buyout", "demerger", "deal", "divest"],
    "Regulation": ["sebi", "regulator", "regulatory", "compliance", "norms", "notice", "show-cause", "ban",
                   "penalty", "gst", "tax"],
    "RBI/Rates": ["rbi", "repo rate", "rate cut", "rate hike", "interest rate", "monetary policy", "mpc",
                  "liquidity", "inflation", "crr", "bps"],
    "Geopolitical": ["war", "conflict", "ceasefire", "tensions", "sanctions", "geopolitical", "border", "tariff",
                     "global selloff"],
    "Legal/Fraud": ["fraud", "court", "lawsuit", "litigation", "enforcement directorate", "scam",
                    "money laundering", "cbi", "allegations", "arrest", "probe"],
    "Management": ["ceo", "cfo", "appoints", "resigns", "resignation", "chairman", "governance", "board"],
}
# tie-break order when two types match equally often
_PRIORITY = ["Earnings", "M&A", "Legal/Fraud", "Regulation", "Management", "RBI/Rates", "Geopolitical"]
_PATTERNS = {t: [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in kws] for t, kws in _KEYWORDS.items()}


def classify_event(headline: str) -> str:
    scores = {t: sum(1 for p in pats if p.search(headline)) for t, pats in _PATTERNS.items()}
    best = max(scores.values())
    if best == 0:
        return "Other"
    return next(t for t in _PRIORITY if scores[t] == best)
