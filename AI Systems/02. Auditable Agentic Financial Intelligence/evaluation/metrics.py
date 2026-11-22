from app.schemas.models import EvaluationResult, VerificationResult

def evaluate(verification: VerificationResult, claims: list[dict], latency_ms: float, estimated_cost_usd: float=0.0) -> EvaluationResult:
    numeric=[c for c in claims if c.get('kind') in ('internal_metric','comparison')]
    bad={i.claim_id for i in verification.issues}
    numerical_accuracy=(sum(c['claim_id'] not in bad for c in numeric)/len(numeric)) if numeric else 1.0
    external=[c for c in claims if c.get('kind') in ('competitor_evidence','comparison')]
    citation_support=(sum(bool(c.get('evidence_ids')) and c['claim_id'] not in bad for c in external)/len(external)) if external else 1.0
    return EvaluationResult(numerical_accuracy=numerical_accuracy,citation_support_rate=citation_support,unsupported_claim_rate=verification.unsupported_claim_rate,latency_ms=latency_ms,estimated_cost_usd=estimated_cost_usd)
