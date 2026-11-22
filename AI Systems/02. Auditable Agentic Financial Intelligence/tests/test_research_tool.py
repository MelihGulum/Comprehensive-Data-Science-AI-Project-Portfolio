from app.tools.research_tool import FixtureProvider, research_company

RECORDS = {
    "Rival Corp": [{
        "title": "Rival Corp FY2025 Results",
        "url": "https://example.com/rival-fy2025",
        "published_date": "2026-02-10",
        "content": "In FY2025 revenue grew 12.5%. Gross margin was 42.0%. Net margin was 9.5%.",
    }]
}

def test_research_builds_traceable_evidence():
    result = research_company("Rival Corp", FixtureProvider(RECORDS))
    assert len(result.sources) == 1
    assert len(result.evidence) == 3
    growth = next(e for e in result.evidence if e.metric == "revenue_growth")
    assert growth.value == 0.125
    assert growth.period == "FY2025"
    assert str(growth.source_url) == "https://example.com/rival-fy2025"
    assert growth.source_title == "Rival Corp FY2025 Results"

def test_missing_url_is_rejected():
    provider = FixtureProvider({"Bad Corp": [{"title": "No URL", "content": "Revenue grew 10%."}]})
    result = research_company("Bad Corp", provider)
    assert result.sources == []
    assert result.evidence == []
    assert result.warnings
