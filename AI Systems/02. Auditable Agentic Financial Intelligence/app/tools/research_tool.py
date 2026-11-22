"""Competitor research with traceable source objects.

The default provider is Tavily when TAVILY_API_KEY is present. A deterministic
fixture provider is included for tests/demo runs. Research never performs
financial arithmetic; it only returns sourced evidence candidates.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Protocol

from app.schemas.models import Evidence, ResearchResult, ResearchSource


class SearchProvider(Protocol):
    name: str
    def search(self, query: str, max_results: int = 5) -> list[dict]: ...


class TavilyProvider:
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not configured")
        from tavily import TavilyClient
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )
        return response.get("results", [])


class FixtureProvider:
    """Offline provider used by tests and the bundled demo."""
    name = "fixture"
    def __init__(self, records: dict[str, list[dict]]):
        self.records = records
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        company = query.split(" annual revenue")[0].strip()
        return self.records.get(company, [])[:max_results]


def _extract_evidence(company: str, source: ResearchSource) -> list[Evidence]:
    """Conservative extraction from snippets; only explicit metric/value pairs.

    This is deliberately narrow in v0.2. LLM extraction comes later behind the
    same schema, while this parser keeps tests deterministic and auditable.
    """
    text = source.snippet.replace(",", "")
    patterns = [
        ("revenue_growth", r"revenue\s+(?:grew|growth)\s+(?:by\s+)?(-?\d+(?:\.\d+)?)\s*%"),
        ("gross_margin", r"gross\s+(?:profit\s+)?margin\s+(?:was|of|:)\s*(-?\d+(?:\.\d+)?)\s*%"),
        ("net_margin", r"net\s+(?:profit\s+)?margin\s+(?:was|of|:)\s*(-?\d+(?:\.\d+)?)\s*%"),
    ]
    period_match = re.search(r"\b(FY\s?20\d{2}|20\d{2})\b", text, re.I)
    period = period_match.group(1).upper().replace(" ", "") if period_match else None
    out: list[Evidence] = []
    for metric, pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = float(match.group(1)) / 100.0
        out.append(Evidence(
            company=company,
            claim=f"{company} {metric.replace('_', ' ')}: {match.group(1)}%" + (f" ({period})" if period else ""),
            metric=metric,
            value=value,
            period=period,
            source_url=source.url,
            source_title=source.title,
            source_date=source.published_at,
            retrieved_at=source.retrieved_at,
            confidence="high" if period else "medium",
            raw_snippet=source.snippet,
        ))
    return out


def research_company(company: str, provider: SearchProvider, max_results: int = 5) -> ResearchResult:
    query = f"{company} annual revenue growth gross margin net margin annual report"
    retrieved = datetime.now(timezone.utc)
    sources: list[ResearchSource] = []
    warnings: list[str] = []
    for item in provider.search(query, max_results=max_results):
        url = item.get("url")
        title = item.get("title") or "Untitled source"
        if not url:
            warnings.append(f"Skipped source without URL: {title}")
            continue
        try:
            sources.append(ResearchSource(
                title=title,
                url=url,
                published_at=item.get("published_date") or item.get("date"),
                retrieved_at=retrieved,
                snippet=item.get("content") or item.get("snippet") or "",
                provider=provider.name,
            ))
        except Exception as exc:
            warnings.append(f"Skipped invalid source {title!r}: {exc}")

    evidence = [e for s in sources for e in _extract_evidence(company, s)]
    if not sources:
        warnings.append("No valid sources found")
    if sources and not evidence:
        warnings.append("Sources found, but no explicit supported metric could be extracted")
    return ResearchResult(company=company, query=query, sources=sources, evidence=evidence, warnings=warnings)


def default_provider() -> SearchProvider:
    return TavilyProvider()
