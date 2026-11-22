from __future__ import annotations
from math import isclose
from app.schemas.models import AnalysisResult, VerificationIssue, VerificationResult

TOL = 1e-9

def verify(analysis: AnalysisResult, metrics: list[dict], evidence: list[dict]) -> VerificationResult:
    issues=[]; supported=0
    metric_map={(str(m['year']), k): m.get(k) for m in metrics for k in ('revenue_growth','gross_margin','net_margin')}
    claims=analysis.internal_claims+analysis.competitor_claims+analysis.comparison_claims
    for c in claims:
        ok=True
        if c.kind == 'internal_metric':
            expected=metric_map.get((c.period or '', c.metric or ''))
            # Canonical metric unit is a decimal ratio: 0.469 == 46.9%.
            # These bounds are deliberately broad; they catch scale/unit bugs, not business outliers.
            if c.value is not None:
                sanity_bounds = {
                    'gross_margin': (-1.0, 2.0),
                    'net_margin': (-5.0, 2.0),
                    'revenue_growth': (-1.0, 10.0),
                }
                bounds = sanity_bounds.get(c.metric or '')
                if bounds and not (bounds[0] <= float(c.value) <= bounds[1]):
                    issues.append(VerificationIssue(
                        claim_id=c.claim_id, code='sanity_check_failed',
                        message=f'{c.metric}={c.value} is outside broad ratio bounds {bounds}; possible percent scaling/unit error.'
                    )); ok=False
            if not c.calculation_ref or expected is None:
                issues.append(VerificationIssue(claim_id=c.claim_id, code='missing_calculation', message='No deterministic calculation supports this claim.')); ok=False
            elif c.value is None or not isclose(c.value, expected, rel_tol=TOL, abs_tol=TOL):
                issues.append(VerificationIssue(claim_id=c.claim_id, code='calculation_mismatch', message='Claim value does not match recalculated metric.')); ok=False
        elif c.kind == 'competitor_evidence':
            if not c.evidence_ids:
                issues.append(VerificationIssue(claim_id=c.claim_id, code='missing_evidence', message='Competitor claim has no evidence reference.')); ok=False
            else:
                for eid in c.evidence_ids:
                    try: ev=evidence[int(eid.split(':')[1])]
                    except (ValueError, IndexError, KeyError):
                        issues.append(VerificationIssue(claim_id=c.claim_id, code='missing_evidence', message=f'Invalid evidence reference: {eid}')); ok=False; continue
                    if ev.get('metric') != c.metric or ev.get('value') is None or c.value is None or not isclose(float(ev['value']), c.value, rel_tol=TOL, abs_tol=TOL):
                        issues.append(VerificationIssue(claim_id=c.claim_id, code='evidence_mismatch', message=f'Claim does not match {eid}.')); ok=False
        elif c.kind == 'comparison':
            if not c.evidence_ids or not c.calculation_ref:
                issues.append(VerificationIssue(claim_id=c.claim_id, code='unsupported_claim', message='Comparison needs both internal calculation and competitor evidence.')); ok=False
            else:
                try: ev=evidence[int(c.evidence_ids[0].split(':')[1])]
                except (ValueError, IndexError, KeyError): ev=None
                own=metric_map.get((c.period or '', c.metric or ''))
                if ev is None or own is None or ev.get('value') is None or c.value is None or not isclose(c.value, own-float(ev['value']), rel_tol=TOL, abs_tol=TOL):
                    issues.append(VerificationIssue(claim_id=c.claim_id, code='evidence_mismatch', message='Comparison delta cannot be reproduced.')); ok=False
        if ok: supported += 1
    rate=(len(claims)-supported)/len(claims) if claims else 0.0
    return VerificationResult(passed=not issues, checked_claims=len(claims), supported_claims=supported, issues=issues, unsupported_claim_rate=rate)

def revise(analysis: AnalysisResult, verification: VerificationResult) -> AnalysisResult:
    """Bounded repair: remove claims that failed verification; never invent replacements."""
    bad={i.claim_id for i in verification.issues}
    analysis.internal_claims=[c for c in analysis.internal_claims if c.claim_id not in bad]
    analysis.competitor_claims=[c for c in analysis.competitor_claims if c.claim_id not in bad]
    analysis.comparison_claims=[c for c in analysis.comparison_claims if c.claim_id not in bad]
    if bad:
        analysis.warnings.append(f"Verifier removed {len(bad)} unsupported claim(s).")
        valid_text={c.text for c in analysis.internal_claims+analysis.comparison_claims}
        analysis.executive_summary=[x for x in analysis.executive_summary if x in valid_text]
    return analysis
