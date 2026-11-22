from app.agents.analyst import analyze
from app.agents.reporter import build_report

def test_analysis_only_compares_same_period():
    metrics = [{"year": 2025, "revenue_growth": .10, "gross_margin": .40, "net_margin": .12}]
    evidence = [{"company":"Rival", "claim":"Rival gross margin: 35.0% (FY2025)", "metric":"gross_margin",
                 "value":.35, "period":"FY2025", "source_url":"https://example.com/a", "source_title":"Results"}]
    result = analyze("Acme", metrics, evidence)
    assert len(result.comparison_claims) == 1
    assert "5.0 percentage points higher" in result.comparison_claims[0].text
    assert result.comparison_claims[0].evidence_ids == ["evidence:0"]

def test_mismatched_period_is_not_compared():
    metrics = [{"year": 2025, "revenue_growth": .10, "gross_margin": .40, "net_margin": .12}]
    evidence = [{"company":"Rival", "claim":"Rival gross margin: 35.0% (FY2024)", "metric":"gross_margin",
                 "value":.35, "period":"FY2024", "source_url":"https://example.com/a", "source_title":"Results"}]
    result = analyze("Acme", metrics, evidence)
    assert not result.comparison_claims

def test_report_preserves_source_link_and_audit_counts():
    metrics = [{"year": 2025, "revenue_growth": .10, "gross_margin": .40, "net_margin": .12}]
    evidence = [{"company":"Rival", "claim":"Rival gross margin: 35.0% (FY2025)", "metric":"gross_margin",
                 "value":.35, "period":"FY2025", "source_url":"https://example.com/a", "source_title":"Results"}]
    analysis = analyze("Acme", metrics, evidence)
    report = build_report("Acme", analysis, evidence)
    assert "https://example.com/a" in report.markdown
    assert report.claim_count == report.supported_claim_count
