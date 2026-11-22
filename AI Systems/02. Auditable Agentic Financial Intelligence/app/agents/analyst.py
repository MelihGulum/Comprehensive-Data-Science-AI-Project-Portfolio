"""Evidence-grounded analysis.

v0.3 intentionally keeps narrative generation deterministic. This makes the
claim/evidence contract testable before an LLM is allowed to paraphrase it.
A later provider can consume AnalysisResult without changing provenance rules.
"""
from __future__ import annotations
from app.schemas.models import AnalysisResult, Claim

LABELS = {
    "revenue_growth": "revenue growth",
    "gross_margin": "gross margin",
    "net_margin": "net margin",
}

def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"

def analyze(company: str, metrics: list[dict], evidence: list[dict]) -> AnalysisResult:
    internal: list[Claim] = []
    competitor: list[Claim] = []
    comparisons: list[Claim] = []
    warnings: list[str] = []

    for row in sorted(metrics, key=lambda x: x["year"]):
        for metric in ("revenue_growth", "gross_margin", "net_margin"):
            value = row.get(metric)
            if value is None:
                continue
            internal.append(Claim(
                claim_id=f"internal:{row['year']}:{metric}",
                text=f"{company} {LABELS[metric]} was {_pct(value)} in {row['year']}.",
                kind="internal_metric", company=company, metric=metric,
                period=str(row["year"]), value=value,
                calculation_ref=f"metrics[{row['year']}].{metric}",
            ))

    for i, ev in enumerate(evidence):
        if ev.get("value") is None or not ev.get("metric"):
            continue
        eid = f"evidence:{i}"
        competitor.append(Claim(
            claim_id=f"competitor:{i}", text=ev["claim"],
            kind="competitor_evidence", company=ev["company"],
            metric=ev.get("metric"), period=ev.get("period"), value=ev.get("value"),
            evidence_ids=[eid], support="supported",
        ))

    if metrics:
        latest = max(metrics, key=lambda x: x["year"])
        latest_year = str(latest["year"])
        for i, ev in enumerate(evidence):
            metric, value = ev.get("metric"), ev.get("value")
            if metric not in LABELS or value is None or latest.get(metric) is None:
                continue
            # Compare only when the competitor evidence clearly names the same year.
            ev_period = (ev.get("period") or "").replace("FY", "")
            if ev_period != latest_year:
                continue
            own = latest[metric]
            delta = own - value
            direction = "higher" if delta > 0 else "lower" if delta < 0 else "equal to"
            if direction == "equal to":
                text = f"{company}'s {LABELS[metric]} matched {ev['company']} at {_pct(own)} in {latest_year}."
            else:
                text = (f"{company}'s {LABELS[metric]} was {abs(delta)*100:.1f} percentage points "
                        f"{direction} than {ev['company']} in {latest_year}.")
            comparisons.append(Claim(
                claim_id=f"comparison:{i}:{metric}", text=text, kind="comparison",
                company=company, metric=metric, period=latest_year, value=delta,
                evidence_ids=[f"evidence:{i}"],
                calculation_ref=f"metrics[{latest_year}].{metric}", support="supported",
            ))

    if evidence and not comparisons:
        warnings.append("Competitor evidence exists, but no same-period comparison could be made safely.")
    if not evidence:
        warnings.append("No competitor evidence was available; competitor comparisons were omitted.")

    latest_claims = [c.text for c in internal if c.period == (str(max(m['year'] for m in metrics)) if metrics else None)]
    summary = latest_claims[:3]
    if comparisons:
        summary.append(comparisons[0].text)
    return AnalysisResult(
        executive_summary=summary, internal_claims=internal,
        competitor_claims=competitor, comparison_claims=comparisons, warnings=warnings,
    )
