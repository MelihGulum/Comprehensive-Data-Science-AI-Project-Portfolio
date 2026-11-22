from app.agents.llm_analyst import enrich_with_llm
from app.agents.analyst import analyze
from app.services.llm import FixtureLLM


def test_structured_insight_must_reference_real_claims():
    a = analyze("Acme", [
        {"year": 2024, "revenue_growth": None, "gross_margin": .40, "net_margin": .10},
        {"year": 2025, "revenue_growth": .10, "gross_margin": .45, "net_margin": .12},
    ], [])
    valid = a.internal_claims[-1].claim_id
    llm = FixtureLLM({
        "executive_analysis": "Profitability improved across the observed period.",
        "insights": [{
            "category": "profitability",
            "title": "Profitability improved",
            "narrative": "The latest profitability metric is stronger.",
            "claim_ids": [valid],
            "evidence_status": "sufficient",
        }],
        "claim_rewrites": {},
    })
    out, _ = enrich_with_llm(a, llm)
    assert out.executive_analysis
    assert len(out.insights) == 1
    assert out.insights[0].claim_ids == [valid]


def test_fake_claim_id_cannot_support_insight():
    a = analyze("Acme", [
        {"year": 2025, "revenue_growth": .10, "gross_margin": .45, "net_margin": .12},
    ], [])
    llm = FixtureLLM({
        "executive_analysis": "",
        "insights": [{
            "category": "growth",
            "title": "Invented",
            "narrative": "Unsupported.",
            "claim_ids": ["fake:id"],
            "evidence_status": "sufficient",
        }],
        "claim_rewrites": {},
    })
    out, _ = enrich_with_llm(a, llm)
    assert out.insights == []


def test_peer_positioning_requires_peer_evidence():
    a = analyze("Acme", [
        {"year": 2025, "revenue_growth": .10, "gross_margin": .45, "net_margin": .12},
    ], [])
    internal_id = a.internal_claims[0].claim_id
    llm = FixtureLLM({
        "executive_analysis": "",
        "insights": [{
            "category": "peer_positioning",
            "title": "Peer position",
            "narrative": "A peer conclusion cannot be supported.",
            "claim_ids": [internal_id],
            "evidence_status": "sufficient",
        }],
        "claim_rewrites": {},
    })
    out, _ = enrich_with_llm(a, llm)
    assert len(out.insights) == 1
    assert out.insights[0].evidence_status == "insufficient"
    assert out.insights[0].claim_ids == []
