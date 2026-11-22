from __future__ import annotations

from app.schemas.models import AnalysisResult, Insight
from app.services.llm import LLMClient, usage_dict

SYSTEM = """You are a grounded financial analysis agent.

Your job is NOT to restate every metric. Produce higher-level interpretation using ONLY
the supplied structured claims. The supplied claims are the complete evidence boundary.

Hard rules:
- Never invent a number, period, company, cause, forecast, recommendation, or external fact.
- Never calculate a new number that is not already represented by the supplied claims.
- Do not infer causality (for example, 'because of pricing') unless an input claim explicitly says it.
- Peer positioning is allowed only when supplied competitor/comparison claims support it.
- If peer evidence is absent or insufficient, mark peer_positioning as evidence_status="insufficient"
  and say that a peer conclusion cannot be made from the available evidence.
- Prefer trends across multiple periods over single-year restatements.
- Distinguish observed facts from interpretation.
- Keep language concise, neutral, and suitable for an investment/finance workspace.

Return JSON exactly in this shape:
{
  "executive_analysis": "2-4 sentence synthesis, or empty string if evidence is insufficient",
  "insights": [
    {
      "category": "profitability|growth|peer_positioning|watch_item|other",
      "title": "short analytical title",
      "narrative": "1-3 sentence interpretation",
      "claim_ids": ["only IDs from supplied claims"],
      "evidence_status": "sufficient|insufficient"
    }
  ],
  "claim_rewrites": {
    "existing_claim_id": "optional concise neutral paraphrase preserving all facts"
  }
}

Generate at most 4 insights. Do not return an executive_summary list.
"""


def _safe_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def enrich_with_llm(analysis: AnalysisResult, llm: LLMClient):
    claims = analysis.internal_claims + analysis.competitor_claims + analysis.comparison_claims
    by_id = {c.claim_id: c for c in claims}

    payload = {
        "claims": [c.model_dump(mode="json") for c in claims],
        "availability": {
            "internal_claims": len(analysis.internal_claims),
            "competitor_claims": len(analysis.competitor_claims),
            "comparison_claims": len(analysis.comparison_claims),
        },
    }
    raw, usage = llm.generate_json(SYSTEM, payload)
    raw = raw if isinstance(raw, dict) else {}

    # LLM may improve prose, but structured values/provenance never move.
    rewrites = raw.get("claim_rewrites", {})
    if isinstance(rewrites, dict):
        for cid, text in rewrites.items():
            if cid in by_id and isinstance(text, str) and text.strip():
                by_id[cid].text = text.strip()

    insights: list[Insight] = []
    raw_insights = raw.get("insights", [])
    if isinstance(raw_insights, list):
        for idx, item in enumerate(raw_insights[:4]):
            if not isinstance(item, dict):
                continue
            category = item.get("category", "other")
            if category not in {"profitability", "growth", "peer_positioning", "watch_item", "other"}:
                category = "other"

            requested_ids = item.get("claim_ids", [])
            valid_ids = [cid for cid in requested_ids if isinstance(cid, str) and cid in by_id]

            status = item.get("evidence_status", "sufficient")
            if status not in {"sufficient", "insufficient"}:
                status = "sufficient"

            # A "sufficient" insight must point to actual supplied claims.
            if status == "sufficient" and not valid_ids:
                continue

            # Peer conclusions require competitor/comparison provenance.
            if category == "peer_positioning":
                peer_ids = {
                    c.claim_id
                    for c in analysis.competitor_claims + analysis.comparison_claims
                }
                if not any(cid in peer_ids for cid in valid_ids):
                    status = "insufficient"
                    valid_ids = []

            title = _safe_text(item.get("title"))
            narrative = _safe_text(item.get("narrative"))
            if not title or not narrative:
                continue

            insights.append(Insight(
                insight_id=f"insight:{idx}",
                category=category,
                title=title,
                narrative=narrative,
                claim_ids=valid_ids,
                evidence_status=status,
            ))

    executive_analysis = _safe_text(raw.get("executive_analysis")) or None

    # Backward-compatible summary is now clean prose, not a Python-list UI artifact.
    if executive_analysis:
        analysis.executive_summary = [executive_analysis]
    elif insights:
        analysis.executive_summary = [x.narrative for x in insights if x.evidence_status == "sufficient"][:3]

    analysis.executive_analysis = executive_analysis
    analysis.insights = insights
    return analysis, usage_dict(usage)
