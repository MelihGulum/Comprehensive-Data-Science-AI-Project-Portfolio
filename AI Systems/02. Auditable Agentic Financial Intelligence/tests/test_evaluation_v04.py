from app.schemas.models import VerificationResult
from evaluation.metrics import evaluate

def test_eval_perfect_verified_run():
    v=VerificationResult(passed=True,checked_claims=1,supported_claims=1,unsupported_claim_rate=0)
    r=evaluate(v,[{'claim_id':'x','kind':'internal_metric'}],123)
    assert r.numerical_accuracy==1 and r.unsupported_claim_rate==0 and r.latency_ms==123
