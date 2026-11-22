from __future__ import annotations
from datetime import datetime, timezone
from app.schemas.models import AnalysisResult, ReportArtifact

def build_report(company: str, analysis: AnalysisResult, evidence: list[dict], verification=None) -> ReportArtifact:
    claims = analysis.internal_claims + analysis.competitor_claims + analysis.comparison_claims
    lines = [f"# Financial Analysis — {company}", "", "## Executive summary"]
    lines += [f"- {x}" for x in analysis.executive_summary] or ["- No supported summary claims available."]
    lines += ["", "## Company performance"]
    lines += [f"- {c.text} `[{c.calculation_ref}]`" for c in analysis.internal_claims]
    lines += ["", "## Competitor evidence"]
    if analysis.competitor_claims:
        for c in analysis.competitor_claims:
            idx = int(c.evidence_ids[0].split(":")[1])
            ev = evidence[idx]
            lines.append(f"- {c.text} [source]({ev['source_url']})")
    else:
        lines.append("- No supported competitor evidence available.")
    lines += ["", "## Comparisons"]
    lines += [f"- {c.text}" for c in analysis.comparison_claims] or ["- No safe same-period comparisons available."]
    if analysis.warnings:
        lines += ["", "## Limitations"] + [f"- {w}" for w in analysis.warnings]
    lines += ["", "## Audit summary",
              f"- Verification: {'PASS' if verification and verification.passed else 'FAIL'}",
              f"- Unsupported claim rate: {verification.unsupported_claim_rate:.1%}" if verification else "- Unsupported claim rate: not checked",
              f"- Structured claims: {len(claims)}",
              f"- Supported claims: {sum(c.support == 'supported' for c in claims)}",
              f"- Evidence objects: {len(evidence)}"]
    return ReportArtifact(
        company=company, generated_at=datetime.now(timezone.utc), markdown="\n".join(lines),
        claim_count=len(claims), supported_claim_count=sum(c.support == "supported" for c in claims),
        evidence_count=len(evidence),
    )
